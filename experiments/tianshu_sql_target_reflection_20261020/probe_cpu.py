# SPDX-License-Identifier: Apache-2.0
"""Real read-only tuning DB snapshot: exact schema/configuration parity and reflection latency."""
import ast,gc,hashlib,json,os,sqlite3,statistics,sys,time
from pathlib import Path
import sqlalchemy as sa
import sqlalchemy.ext.automap
BASE=Path(__file__).resolve().parent
LIVE=Path('/workspace/FlagGems/src/flag_gems/utils/models/sql.py')
root=Path('/workspace/benchmark-results/20261020_sql_target_reflection_CPU_v2_tianshu')
assert not root.exists();root.mkdir(parents=True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    tree=ast.parse(Path(path).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SQLPersistantModel')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='build_sql_model_by_db');fn.decorator_list=[]
    namespace={'sqlalchemy':sa};exec('from __future__ import annotations\n'+ast.unparse(fn),namespace)
    cls.body=[n for n in cls.body if n is not fn]
    return namespace[fn.name],ast.dump(tree,include_attributes=False)
native,native_rest=load(BASE/'native_sql.py');candidate,candidate_rest=load(BASE/'candidate_sql.py')
assert native_rest==candidate_rest and sha(LIVE)==sha(BASE/'native_sql.py')
source_sha={str(p):sha(p) for p in [LIVE,BASE/'native_sql.py',BASE/'candidate_sql.py',BASE/'verify_cpu.py',BASE/'run_after_gpu.sh',BASE/'README.md',Path(__file__).resolve()]}
result=dict(complete=False,phase='regression',runner_pid=os.getpid(),sources=source_sha,sqlalchemy_version=sa.__version__,
    upstream_commit='8eb57f17d5a3a93f7399af440d8dc319b0916c00',only_reflection_method_AST_changed=True,
    no_GPU_or_service_started=True,live_FlagGems_source_not_modified=True,live_database_never_written=True,databases=[],timing_rounds=[])
def save():
    tmp=root/'result.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(root/'result.json')
def watched(engine,fn,name):
    statements=[]
    def capture(conn,cursor,statement,parameters,context,many):statements.append(statement)
    sa.event.listen(engine,'before_cursor_execute',capture)
    try:value=fn(name,engine)
    finally:sa.event.remove(engine,'before_cursor_execute',capture)
    return value,statements
def schema(table):return [(c.name,str(c.type),c.nullable,c.primary_key) for c in table.columns]
def rows_of(engine,table):
    with engine.connect() as connection:return [list(row) for row in connection.execute(sa.select(table).order_by(*table.primary_key.columns))]
save()
try:
    engine=sa.create_engine('sqlite:///:memory:');metadata=sa.MetaData()
    target=sa.Table('kernel-tune-with-hyphens',metadata,sa.Column('key_0',sa.BigInteger,primary_key=True),sa.Column('key_1',sa.String,primary_key=True),sa.Column('BLOCK_M',sa.Integer),sa.Column('flag',sa.Boolean),sa.Column('optional',sa.Float,nullable=True))
    for i in range(12):sa.Table(f'unrelated_{i}',metadata,sa.Column('id',sa.Integer,primary_key=True))
    metadata.create_all(engine)
    with engine.begin() as connection:connection.execute(target.insert(),dict(key_0=2**40,key_1='bf16',BLOCK_M=128,flag=True,optional=None))
    old=native(target.name,engine);new,statements=watched(engine,candidate,target.name)
    assert schema(old.__table__)==schema(new.__table__) and rows_of(engine,new.__table__)==[[2**40,'bf16',128,True,None]]
    assert not any('PRAGMA' in s and 'unrelated_' in s for s in statements)
    missing,statements=watched(engine,candidate,'missing');assert missing is None and not any('PRAGMA' in s for s in statements)
    empty=sa.create_engine('sqlite:///:memory:');assert native('absent',empty) is candidate('absent',empty) is None
    with empty.begin() as connection:connection.exec_driver_sql('create table no_primary_key (value integer)')
    assert native('no_primary_key',empty) is candidate('no_primary_key',empty) is None
    result['synthetic_regressions']=dict(composite_primary_key=True,BIGINT_over_32bit=True,boolean_and_NULL=True,hyphenated_table_name=True,unrelated_tables_not_reflected=True,missing_table_no_PRAGMA=True,empty_database=True,unmapped_no_primary_key=True)
    engine.dispose();empty.dispose();result['phase']='actual_database';save()
    source=Path('/root/.flaggems/config_cache/TunedConfig_iluvatar_triton_3_6.db');snapshot=root/source.name
    with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as live:
        with sqlite3.connect(snapshot) as destination:live.backup(destination)
    engine=sa.create_engine('sqlite:///file:'+str(snapshot)+'?mode=ro&uri=true');names=sa.inspect(engine).get_table_names()
    db=dict(source=str(source),snapshot=str(snapshot),snapshot_sha256=sha(snapshot),table_count=len(names),tables=[],database_rows=0)
    result['databases'].append(db);save()
    start=time.perf_counter();reference_model,full_queries=watched(engine,native,names[0]);full_native_ms=(time.perf_counter()-start)*1000
    assert reference_model is not None
    old_models={mapper.local_table.name:mapper.class_ for mapper in reference_model.registry.mappers}
    result["full_native_reflection"]=dict(target=names[0],milliseconds=full_native_ms,SQL_statements=len(full_queries),once_only=True,mapped_tables=len(old_models),timing_scope="Single cold metadata reflection with SQL statement instrumentation")
    save()
    mappable=[]
    for name in names:
        old=old_models.get(name);start=time.perf_counter();new,statements=watched(engine,candidate,name);candidate_ms=(time.perf_counter()-start)*1000;assert (old is None)==(new is None)
        if old is None:db['tables'].append(dict(name=name,mapped=False));continue
        assert schema(old.__table__)==schema(new.__table__)
        old_rows=rows_of(engine,old.__table__);new_rows=rows_of(engine,new.__table__);assert old_rows==new_rows
        payload=json.dumps(new_rows,sort_keys=True,separators=(',',':')).encode()
        db['tables'].append(dict(name=name,mapped=True,schema=schema(new.__table__),primary_key=list(new.__table__.primary_key.columns.keys()),rows=len(new_rows),row_values_sha256=hashlib.sha256(payload).hexdigest(),candidate_reflection_SQL_statements=len(statements),candidate_instrumented_reflection_ms=candidate_ms))
        db['database_rows']+=len(new_rows);mappable.append(name)
        if len(db['tables'])%32==0:save()
    assert mappable and db['database_rows']>0
    selected=sorted(set([mappable[0],mappable[len(mappable)//2],mappable[-1]]))
    result['phase']='timing';save()
    for name in selected:
        for index in range(7):
            start=time.perf_counter();model=candidate(name,engine);elapsed=(time.perf_counter()-start)*1000
            assert model is not None
            result['timing_rounds'].append(dict(table=name,round=index,mode='candidate',milliseconds=elapsed));save()
            del model;gc.collect()
    db['timed_tables']=selected;engine.dispose()
    assert sha(snapshot)==db['snapshot_sha256'] and all(sha(path)==value for path,value in source_sha.items())
    result.update(complete=True,phase='complete',steady_state_GPU_throughput_not_measured=True,adopted=False);save()
except BaseException as error:result.update(phase='failed',failure=repr(error));save();raise
