# SPDX-License-Identifier: Apache-2.0
"""Execute actual native logger.debug statements and its once filter without importing GPU libraries."""
import ast,hashlib,json,logging,statistics,time
from pathlib import Path
from level import apply
BASE=Path(__file__).resolve().parent
ROOT=Path('/workspace/benchmark-results/20261021_debug_log_level_CPU_tianshu')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not ROOT.exists();ROOT.mkdir()
    logging_source=Path('/usr/local/lib/python3.12/site-packages/flag_gems/logging_utils.py')
    tree=ast.parse(logging_source.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LogOncePerLocationFilter')
    namespace=dict(logging=logging);exec(compile(ast.Module(body=[node],type_ignores=[]),str(logging_source),'exec'),namespace)
    Once=namespace['LogOncePerLocationFilter'];plans=[];sources=[Path(__file__),BASE/'level.py',logging_source]
    for op,name,repeat in (('add','add',2),('alias','alias',1),('copy','copy_',3),('fill','fill_scalar_',4),('index','index',1),('sub','sub',1),('to','to_copy',1)):
        path=logging_source.parents[0]/'ops'/f'{op}.py';sources.append(path)
        fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        call=next(n for n in fn.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='logger.debug')
        assert len(call.value.args)==1 and isinstance(call.value.args[0],ast.Constant)
        code=compile(ast.Module(body=[call],type_ignores=[]),str(path),'exec');plans.append((code,repeat))
    assert sum(r for c,r in plans)==13
    def fixture(tag):
        logger=logging.Logger('flag_gems_fixture_'+tag,level=logging.DEBUG);logger.propagate=False
        handler=logging.FileHandler(ROOT/f'{tag}.log',mode='w');handler._flaggems_owned=True;handler.addFilter(Once())
        handler.setFormatter(logging.Formatter('%(levelname)s %(message)s'));logger.addHandler(handler)
        return logger,handler
    def work(logger):
        ns=dict(logger=logger)
        for code,repeats in plans:
            for _ in range(repeats):exec(code,ns)
    def high(logger):
        logger.info('INFO must stay')
        logger.warning('WARNING must stay')
        logger.error('ERROR must stay')
        logger.critical('CRITICAL must stay')
    original_factory=logging.getLogRecordFactory();checks=[]
    try:
        for mode in ('control','candidate'):
            logger,handler=fixture(mode);proof=apply(logger,mode);counts={}
            def factory(*args,**kwargs):
                row=original_factory(*args,**kwargs);counts[row.levelname]=counts.get(row.levelname,0)+1;return row
            logging.setLogRecordFactory(factory)
            for _ in range(2):work(logger)
            high(logger);handler.flush();logging.setLogRecordFactory(original_factory)
            text=(ROOT/f'{mode}.log').read_text();handler.close()
            assert counts.get('DEBUG',0)==(26 if mode=='control' else 0)
            assert text.count('DEBUG ')==(7 if mode=='control' else 0)
            assert all(counts[level]==1 and f'{level} ' in text for level in ('INFO','WARNING','ERROR','CRITICAL'))
            checks.append(dict(mode=mode,records_constructed=counts,debug_lines_written=text.count('DEBUG '),proof=proof))
        timings=[]
        for trial in range(7):
            values={}
            for mode in (('control','candidate') if trial%2==0 else ('candidate','control')):
                logger,handler=fixture(f'timing_{trial}_{mode}');apply(logger,mode);work(logger)
                start=time.perf_counter_ns()
                for _ in range(200):work(logger)
                values[mode]=(time.perf_counter_ns()-start)/200/1000;handler.close()
            timings.append(values)
    finally:logging.setLogRecordFactory(original_factory)
    gate=dict(complete=True,CPU_only=True,no_Torch_or_GPU_libraries_imported=True,actual_native_debug_statement_sites=7,
        actual_native_debug_calls_per_iteration=13,actual_native_once_filter_AST_executed=True,checks=checks,
        timings_us=timings,control_median_us=statistics.median(r['control'] for r in timings),candidate_median_us=statistics.median(r['candidate'] for r in timings),
        timing_scope='CPU 13 real native DEBUG statements, 7 alternating x200, RecordFactory instrumentation disabled for timing; not model score',
        all_INFO_WARNING_ERROR_CRITICAL_records_and_text_preserved=True,sources={str(p.resolve()):sha(p) for p in sources})
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k not in ('sources','checks','timings_us')}))
if __name__=='__main__':main()
