# SPDX-License-Identifier: Apache-2.0
"""Read-only CPU diagnosis of completed original-script N64 cache reuse."""
import hashlib,json,re
from pathlib import Path
BASE=Path(__file__).resolve().parent
PAIR=Path('/workspace/benchmark-results/20261024_W4_O64_affine_screen_pair_tianshu')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def counter(p,name):
    rows=[float(line.rsplit(' ',1)[1]) for line in p.read_text().splitlines() if line.startswith('vllm:'+name+'{')]
    assert len(rows)==1,(p,name);return rows[0]
def main():
    screen=json.loads((BASE/'SCREEN_GATE_32g.json').read_text());assert screen['complete']
    arms={};hashes={}
    for arm in ('control','candidate'):
        root=PAIR/arm;state=json.loads((root/'result.json').read_text())
        assert state['phase']=='complete' and state['evaluation_exit_code']==0
        before,after=root/'before_metrics.txt',root/'after_metrics.txt'
        queries=counter(after,'prefix_cache_queries_total')-counter(before,'prefix_cache_queries_total')
        hits=counter(after,'prefix_cache_hits_total')-counter(before,'prefix_cache_hits_total')
        assert queries==1048576 and hits==783360
        seeds=re.findall(r'\bseed=(\d+)',(root/'evaluation.log').read_text())
        assert seeds==['0']*4,(arm,seeds)
        arms[arm]=dict(queries=queries,hits=hits,hit_fraction=hits/queries,actual_logged_client_seeds=seeds)
        for p in (before,after,root/'evaluation.log',root/'result.json'):hashes[str(p)]=digest(p)
    gate=dict(complete=True,CPU_only=True,arms=arms,raw_files=hashes,
        original_screen_gate_unchanged_sha256=digest(BASE/'SCREEN_GATE_32g.json'),
        not_representative_of_fresh_4K_inputs=True,performance_adopted=False,
        apparent_paired_change_percent=screen['paired_total_throughput_change_percent'],
        original_stability_gate_passed=screen['meets_1percent_throughput_and_1percent_spread_gate'],
        full_prompt_bodies_not_serialized=True,per_round_hits_not_observed=True,
        source_sha256=digest(Path(__file__)))
    assert not gate['original_stability_gate_passed']
    (BASE/'CACHE_DIAGNOSTIC_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in gate.items() if k!='raw_files'},indent=2))
if __name__=='__main__':main()
