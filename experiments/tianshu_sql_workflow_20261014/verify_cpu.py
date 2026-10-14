# SPDX-License-Identifier: Apache-2.0
"""CPU-only real source/loader composition and configuration row equivalence."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--original-sql',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    source=args.original_sql.resolve()
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='6dc19cb581d70154f14b95ed9112d682c6b5919b6cf3f8dea01a78a75ec0efd7'
    import sqlalchemy as sa
    import sqlalchemy.ext.automap
    import sqlalchemy.orm
    # Execute the original function AST without importing Torch/Triton.
    tree=ast.parse(source.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SQLPersistantModel')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='build_sql_model_by_db')
    method.decorator_list=[]
    method.returns=None
    for argument in method.args.args:argument.annotation=None
    namespace={'sqlalchemy':sa}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),namespace)
    original=namespace['build_sql_model_by_db']
    engine=sa.create_engine('sqlite://')
    metadata=sa.MetaData()
    for i in range(128):
        sa.Table(f'config_{i}',metadata,sa.Column('M',sa.Integer,primary_key=True),
                 sa.Column('dtype',sa.String,primary_key=True),sa.Column('BLOCK_M',sa.Integer),
                 sa.Column('num_warps',sa.Integer))
    metadata.create_all(engine)
    with engine.begin() as connection:
        for i in range(128):
            connection.execute(metadata.tables[f'config_{i}'].insert(),[
                dict(M=1,dtype='bf16',BLOCK_M=16+i,num_warps=4),
                dict(M=64,dtype='bf16',BLOCK_M=64+i,num_warps=8)])
    base=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='tianshu_sql_s8_cpu_',dir='/tmp') as temporary:
        os.environ.update(TIANSHU_SQL_REFLECTION_PROBE='1',TIANSHU_SQL_REFLECTION_RECORD_DIR=temporary,
                          TIANSHU_SEGMENT_POLICY_PROBE='1',TIANSHU_KV16_SERVICE_MODE='candidate')
        previous=list(sys.meta_path)
        hook=runpy.run_path(str(base/'hooks/sitecustomize.py'))
        parents=[]

        class Parent:
            def exec_module(self,module):
                parents.append(True)
                module.SQLPersistantModel=type('SQLPersistantModel',(),{'build_sql_model_by_db':staticmethod(original)})

        module=SimpleNamespace(__file__=str(source))
        hook['Loader'](Parent()).exec_module(module)
        assert len(parents)==1
        checks=[]
        for name in ('config_0','config_63','config_127','missing'):
            before=original(name,engine)
            after=module.SQLPersistantModel.build_sql_model_by_db(name,engine)
            assert (before is None)==(after is None)
            rows=0
            if before is not None:
                def columns(model):return [(x.name,str(x.type),x.primary_key,x.nullable) for x in model.__table__.columns]
                assert columns(before)==columns(after)
                with engine.connect() as connection:
                    a=connection.execute(sa.select(before.__table__).order_by(*before.__table__.primary_key)).mappings().all()
                    b=connection.execute(sa.select(after.__table__).order_by(*after.__table__.primary_key)).mappings().all()
                    assert [dict(x) for x in a]==[dict(x) for x in b]
                    rows=len(a)
            checks.append(dict(table=name,identical=True,rows=rows))
        actual=json.loads(next(Path(temporary).glob('route_*.json')).read_text())
        assert actual['reflection_scope']=='one_requested_table' and not actual['configuration_values_modified']
        assert any(type(f).__name__=='Finder' for f in sys.meta_path)
        assert 'torch' not in sys.modules and 'vllm' not in sys.modules
        sys.meta_path[:]=previous
    result=dict(complete=True,cpu_only=True,parent_loader_preserved=True,S8_base_source_verified=True,
                original_sql_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                hook_sha256=hashlib.sha256((base/'hooks/sitecustomize.py').read_bytes()).hexdigest(),
                helper_sha256=hashlib.sha256((base/'reflect_one.py').read_bytes()).hexdigest(),
                tables=128,checks=checks,actual_SQL_lookup_record=actual,gpu_executed=False)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
