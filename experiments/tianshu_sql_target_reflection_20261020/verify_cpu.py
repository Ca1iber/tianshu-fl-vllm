# SPDX-License-Identifier: Apache-2.0
"""Independent SQLite row/schema recheck and paired cold-reflection statistics."""
import hashlib,json,sqlite3,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent;root=Path('/workspace/benchmark-results/20261020_sql_target_reflection_CPU_v2_tianshu')
result=json.loads((root/'result.json').read_text());assert result['complete'] and result['phase']=='complete' and result['only_reflection_method_AST_changed']
assert all(result['synthetic_regressions'].values()) and all(result[k] for k in ('no_GPU_or_service_started','live_FlagGems_source_not_modified','live_database_never_written'))
for path,value in result['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==value
assert len(result['databases'])==1;db=result['databases'][0]
assert hashlib.sha256(Path(db['snapshot']).read_bytes()).hexdigest()==db['snapshot_sha256']
count=0
with sqlite3.connect('file:'+db['snapshot']+'?mode=ro',uri=True) as connection:
    names=[row[0] for row in connection.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name")]
    assert len(names)==db['table_count']==len(db['tables'])
    for table in db['tables']:
        name=table['name'].replace('"','""');info=list(connection.execute('PRAGMA table_info("'+name+'")'))
        pk=[row[1] for row in sorted((row for row in info if row[5]),key=lambda row:row[5])]
        if not table['mapped']:assert not pk;continue
        assert pk==table['primary_key'] and [row[1] for row in info]==[col[0] for col in table['schema']]
        order=','.join('"'+key.replace('"','""')+'"' for key in pk)
        rows=[list(row) for row in connection.execute('select * from "'+name+'" order by '+order)]
        # Real tuning cache stores primitive INTEGER/TEXT/FLOAT/NULL values;
        # bool is restored by SQLAlchemy only when declared BOOLEAN.
        for index,column in enumerate(table['schema']):
            if column[1]=='BOOLEAN':
                for row in rows:
                    if row[index] is not None:row[index]=bool(row[index])
        payload=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
        assert len(rows)==table['rows'] and hashlib.sha256(payload).hexdigest()==table['row_values_sha256'];count+=len(rows)
assert count==db['database_rows'];comparisons=[]
for name in db['timed_tables']:
    timings=[row for row in result['timing_rounds'] if row['table']==name];assert len(timings)==7 and sorted(t['round'] for t in timings)==list(range(7))
    assert all(t['mode']=='candidate' and t['milliseconds']>0 for t in timings)
    comparisons.append(dict(table=name,candidate_reflection_median_ms=statistics.median(t['milliseconds'] for t in timings)))
old=result['full_native_reflection'];assert old['once_only'] and old['milliseconds']>0
matched=next(t for t in db['tables'] if t['name']==old['target']);assert matched['mapped']
pair=dict(target=old['target'],native_single_instrumented_ms=old['milliseconds'],candidate_single_instrumented_ms=matched['candidate_instrumented_reflection_ms'],native_SQL_statements=old['SQL_statements'],candidate_SQL_statements=matched['candidate_reflection_SQL_statements'],single_sample_only=True)
gate=dict(complete=True,independently_SQLite_rechecked=True,sources=result['sources'],upstream_commit=result['upstream_commit'],only_reflection_method_AST_changed=True,synthetic_regressions=result['synthetic_regressions'],
    actual_database_tables=db['table_count'],actual_database_rows=count,all_config_values_schema_and_primary_keys_identical=True,source_database=db['source'],snapshot_sha256=db['snapshot_sha256'],
    candidate_repeated_timings=comparisons,native_metadata_reflected_once_for_full_parity=True,single_instrumented_first_target_pair=pair,
    timing_groups=len(result['timing_rounds']),no_GPU_or_service_started=True,live_FlagGems_source_not_modified=True,live_database_never_written=True,steady_state_GPU_throughput_not_measured=True,adopted=False)
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');(BASE/'CPU_ROUNDS_32g.json').write_text(json.dumps(result['timing_rounds'],indent=2)+'\n')
print(json.dumps(gate))
