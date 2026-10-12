# SPDX-License-Identifier: Apache-2.0
"""Process-only M8 projection experiment using the numerically tested kernel."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

BASE=Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE/'tianshu_fa2_tiles_20261012/hooks/sitecustomize.py'))
MM_SHA='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
KERNEL_SHA='86a2244f3fb30a8294d53174bd3aa6080f947b5b80ce43b6212a35b533463d56'


class SmallLoader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=MM_SHA:
            raise RuntimeError('Small GEMM requires the inspected vendor source')
        source=BASE/'tianshu_splitk_20261012/experimental_mm.py'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=KERNEL_SHA:
            raise RuntimeError('Small GEMM kernel differs from120 checked configurations')
        spec=importlib.util.spec_from_file_location('tianshu_small_mm_kernel',source)
        kernel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernel;spec.loader.exec_module(kernel)
        import torch
        original=module._launch_mm
        recorded=set()
        plans={(2560,2048):2,(2048,2048):2,(2048,6144):4}

        def launch(a,b,c,M,N,K):
            splits=plans.get((N,K))
            if (M!=8 or splits is None or a.dtype!=torch.bfloat16 or b.dtype!=torch.bfloat16
                or c.dtype!=torch.bfloat16 or not a.is_contiguous() or not c.is_contiguous()
                or b.stride()!=(1,K) or a.device!=b.device or c.device!=a.device):
                return original(a,b,c,M,N,K)
            config=dict(bm=16,bn=64,bk=64,splits=splits,warps=4,stages=2)
            partial=torch.empty((splits,M,N),device=a.device,dtype=torch.float32)
            # Pass the native preallocated output as the verified kernel's
            # scratch output; no extra output copy or weight layout change.
            kernel.mm(a,b.T,(partial,c),config)
            directory=os.environ.get('TIANSHU_SMALL_GEMM_RECORD_DIR')
            if directory and (M,N,K) not in recorded:
                recorded.add((M,N,K));out=Path(directory);out.mkdir(parents=True,exist_ok=True)
                (out/f'mm_{os.getpid()}_{M}_{N}_{K}.json').write_text(json.dumps({
                    'pid':os.getpid(),'enabled':True,'shape':[M,N,K],'configuration':config,
                    'vendor_source':module.__file__,'vendor_sha256':MM_SHA,
                    'kernel_source':str(source),'kernel_sha256':KERNEL_SHA,
                    'output_copy':False,'weights_reordered':False},indent=2)+'\n')
            return c
        module._launch_mm=launch


class SmallFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        # FlagGems imports the vendor package via sys.path as _iluvatar.
        if fullname not in ('_iluvatar.ops.mm','flag_gems.runtime.backend._iluvatar.ops.mm'):return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None or spec.loader is None:raise RuntimeError('Vendor GEMM module unavailable')
        spec.loader=SmallLoader(spec.loader);return spec


if os.environ.get('TIANSHU_SMALL_GEMM_PROBE')=='1':sys.meta_path.insert(0,SmallFinder())
