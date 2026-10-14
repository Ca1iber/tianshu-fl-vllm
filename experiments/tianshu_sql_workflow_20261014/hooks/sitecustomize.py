# SPDX-License-Identifier: Apache-2.0
"""Process-only target-table SQL reflection on the verified Tianshu profile."""
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import sys
import time


VLLM=Path(__file__).resolve().parents[2]
base=VLLM/'tianshu_kv16_segments_20261014/hooks/sitecustomize.py'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='71b8368cca267314481f11e4efe5556b083ca5e86d6e80380ba4ab8632b54a19'
runpy.run_path(str(base))


class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        sha=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        assert sha=='6dc19cb581d70154f14b95ed9112d682c6b5919b6cf3f8dea01a78a75ec0efd7'
        path=Path(__file__).resolve().parents[1]/'reflect_one.py'
        assert hashlib.sha256(path.read_bytes()).hexdigest()=='4cfc6a34e29e0cc220b83d79898c1da4d986e35d4eb0796c37947b1631064c02'
        helper=runpy.run_path(str(path))['reflect_one'];seen=set()
        def build(name,engine):
            start=time.perf_counter();result=helper(name,engine);elapsed=time.perf_counter()-start
            # One actual lookup record per process, outside any GPU kernel.
            if not seen:
                seen.add(name)
                root=Path(os.environ['TIANSHU_SQL_REFLECTION_RECORD_DIR']);root.mkdir(parents=True,exist_ok=True)
                (root/f'route_{os.getpid()}.json').write_text(json.dumps(dict(
                    original_sql_sha256=sha,helper_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    reflection_scope='one_requested_table',table=name,table_exists=result is not None,
                    elapsed_seconds=elapsed,configuration_values_modified=False),indent=2)+'\n')
            return result
        module.SQLPersistantModel.build_sql_model_by_db=staticmethod(build)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='flag_gems.utils.models.sql':return None
        for finder in tuple(sys.meta_path):
            if finder is self:continue
            method=getattr(finder,'find_spec',None);spec=method(fullname,path,target) if method else None
            if spec is not None:spec.loader=Loader(spec.loader);return spec
        return None


if os.getenv('TIANSHU_SQL_REFLECTION_PROBE')=='1':sys.meta_path.insert(0,Finder())
