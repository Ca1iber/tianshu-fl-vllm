# SPDX-License-Identifier: Apache-2.0
"""Compose S8/SQL with bounded83-point buckets without editing installed files."""
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import sys


BASE=Path(__file__).resolve().parents[2]
runpy.run_path(str(BASE/'tianshu_sql_workflow_20261014/hooks/sitecustomize.py'))


class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        digest=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        assert digest=='966aad18aad80ef488928cafc8672cbfa2d547a832f9b151346f719ea3abc6db'
        helper=Path(__file__).resolve().parents[1]/'extend_buckets.py'
        provenance=runpy.run_path(str(helper))['extend'](module)
        root=Path(os.environ['TIANSHU_EXTENDED_BUCKET_RECORD_DIR'])
        root.mkdir(parents=True,exist_ok=True)
        provenance.update(pid=os.getpid(),original_bucket_source_sha256=digest,
                          helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                          installed_sources_modified=False)
        (root/f'extended_{os.getpid()}.json').write_text(json.dumps(provenance,indent=2)+'\n')


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm_fl.worker.iluvatar_prefill_buckets':return None
        # Preserve the existing configure/warmup/padding tracer loader.
        for finder in tuple(sys.meta_path):
            if finder is self:continue
            method=getattr(finder,'find_spec',None)
            spec=method(fullname,path,target) if method else None
            if spec is not None:
                spec.loader=Loader(spec.loader);return spec
        raise RuntimeError('Original bucket module unavailable')


if os.getenv('TIANSHU_EXTENDED_BUCKET_PROBE')=='1':sys.meta_path.insert(0,Finder())
