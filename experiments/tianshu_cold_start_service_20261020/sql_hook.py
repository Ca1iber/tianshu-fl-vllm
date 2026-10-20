# SPDX-License-Identifier: Apache-2.0
"""Install exact target-table reflection before FlagGems first uses its SQL model."""
from __future__ import annotations
import atexit,functools,hashlib,importlib.abc,importlib.util,json,os,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
sqlalchemy=None
EXPECTED_NATIVE='6dc19cb581d70154f14b95ed9112d682c6b5919b6cf3f8dea01a78a75ec0efd7'

def build_sql_model_by_db(name: str,engine: sqlalchemy.engine.Engine) -> Optional[Type[Base]]:
    if name not in sqlalchemy.inspect(engine).get_table_names():
        return None
    AutoBase: sqlalchemy.ext.automap.AutomapBase = sqlalchemy.ext.automap.automap_base()
    AutoBase.prepare(autoload_with=engine,reflection_options={"only": [name]})
    ModelCls: Optional[Type[Base]] = AutoBase.classes.get(name)
    return ModelCls

def install():
    mode=os.environ['TIANSHU_COLD_START_MODE'];assert mode in ('native','cpu4','gpu4')
    directory=Path(os.environ['TIANSHU_COLD_START_RECORD_DIR'])/'sql'
    def patch(module,preimported=False):
        global sqlalchemy
        sqlalchemy=module.sqlalchemy;source=Path(module.__file__).resolve();digest=hashlib.sha256(source.read_bytes()).hexdigest();assert digest==EXPECTED_NATIVE
        assert not getattr(module.SQLPersistantModel,'_tianshu_cold_sql',False)
        original=module.SQLPersistantModel.build_sql_model_by_db
        record=dict(installed=True,mode=mode,pid=os.getpid(),preimported_module=preimported,original_source=str(source),original_sha256=digest,
            hook_source=str(Path(__file__).resolve()),hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            target_reflection=mode!='native',calls={},total_calls=0,total_seconds=0.,original_class_other_methods_unchanged=True)
        def save():
            directory.mkdir(parents=True,exist_ok=True)
            path=directory/f'reflection_{os.getpid()}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
        @functools.wraps(original)
        def reflected(name,engine):
            if record['pid']!=os.getpid():record.update(inherited_from_pid=record['pid'],pid=os.getpid(),calls={},total_calls=0,total_seconds=0.)
            started=time.perf_counter()
            try:return original(name,engine) if mode=='native' else build_sql_model_by_db(name,engine)
            finally:
                elapsed=time.perf_counter()-started;row=record['calls'].setdefault(name,dict(count=0,seconds=0.))
                row['count']+=1;row['seconds']+=elapsed;record['total_calls']+=1;record['total_seconds']+=elapsed;save()
        module.SQLPersistantModel.build_sql_model_by_db=staticmethod(reflected);module.SQLPersistantModel._tianshu_cold_sql=True
        atexit.register(save);save()
    class Loader(importlib.abc.Loader):
        def __init__(self,original):self.original=original
        def create_module(self,spec):return self.original.create_module(spec)
        def exec_module(self,module):self.original.exec_module(module);patch(module)
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname!='flag_gems.utils.models.sql':return None
            index=sys.meta_path.index(self);sys.meta_path.pop(index)
            try:
                spec=None
                for finder in tuple(sys.meta_path):
                    method=getattr(finder,'find_spec',None)
                    if method is not None:
                        spec=method(fullname,path,target)
                        if spec is not None:break
            finally:sys.meta_path.insert(index,self)
            assert spec is not None and spec.loader is not None;spec.loader=Loader(spec.loader);return spec
    if 'flag_gems.utils.models.sql' in sys.modules:patch(sys.modules['flag_gems.utils.models.sql'],True)
    else:sys.meta_path.insert(0,Finder())
