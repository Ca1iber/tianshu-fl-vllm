# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
path=root/'result.json';data=json.loads(path.read_text())
assert data['complete'] and data['phase']=='complete' and data['all_staged_values_bitwise'] and data['all_guards_and_native_fallback_verified']
assert not data['adopted'] and not data['official_benchmark'] and not data['service_executed']
for name,h in data['frozen_sources'].items():assert sha(name)==sha(root/'source_snapshot'/Path(name).relative_to('/'))==h
assert 'No running processes found' in (root/'hardware_before.txt').read_text()
assert len(data['cases'])==6 and len({c['name'] for c in data['cases']})==6
staged=elements=fallbacks=0
for row in data['cases']:
    assert row['same_Graph_two_phases'] and row['raw_cache_unchanged'] and row['RNG_unchanged']
    assert len(row['checks'])==4
    for check in row['checks']:
        if check['fallback']:
            assert check['mode']=='candidate' and row['pages']>8192 and check['arena_unchanged'];fallbacks+=1
        else:
            assert check['active_cells_bitwise'] and check['scratch_guard_preserved'] and check['distinct_dense_page_ids']
            assert check['unused_pages_and_rows_guarded'] and check['reused_pointers'];staged+=1;elements+=check['checked_elements']
assert fallbacks==2
assert staged==22 and data['scratch_saved_bytes']==384*1024**2 and data['table_saved_bytes']==96*1024
for mode,pages in [('control',32768),('candidate',8192)]:
    row=data['mode_reservations'][mode]
    assert row['capacity_pages']==pages and row['scratch_bytes']==(pages+1)*16384 and row['table_bytes']==pages*4
    assert row['single_reusable_arena'] and row['allocated_after']-row['allocated_before']>=row['scratch_bytes']+row['table_bytes']
data.update(independently_rechecked=True,staging_GPU_checks=staged,fallback_branch_checks=fallbacks,
            active_cells_bitwise_verified=elements,raw_result_path=str(path),raw_result_sha256=sha(path),analyzer_source_sha256=sha(__file__),
            fallback_scope='Original staging returns None without arena access; full attention fallback is inherited unchanged and covered by prior qualified component, not executed by this probe')
(Path(__file__).resolve().parent/'GPU_GATE_32g.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(dict(complete=True,staged_checks=staged,fallbacks=fallbacks,active_cells=elements)))
