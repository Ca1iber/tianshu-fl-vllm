# SPDX-License-Identifier: Apache-2.0
import hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent

def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (BASE/'selected_plan.json').exists();gates={}
 for version in ('','_v2'):
  directory=EXP/f'tianshu_native_int8_attention42{version}_20261022';gate=read(directory/'INDEPENDENT_CPU_GATE_32g.json');assert gate['complete'] and gate['artifacts_checked']==504;raw=Path(gate['raw_directory']);assert sha(raw/'result.json')==gate['raw_result_sha256'];r=read(raw/'result.json');assert r['complete'] and len(r['banks'])==84 and r['all84_bank_hashes_before_after'];gates[version]=(directory,gate,r)
 first=gates[''][2];second=gates['_v2'][2];assert first['banks']==second['banks'];rows=dict(QKV=[128,512,4096],O=[512,4096]);binaries={f'{role}:{case["M"]}':case['compiled'][role]['binary_sha256'] for case in first['cases'] for role in ('QKV','O') if case['M'] in rows[role]};assert len(binaries)==5
 evidence=Path('/tmp/torchinductor_root/b5/cb5sev4w4yksw5yzsdnubu5t2t2vtkvns5apw5gj67rzflghednm.py');text=evidence.read_text();assert 'torch.ops.tianshu_native_down.linear.default' in text and 'extern_kernels.mm' in text and 'buf9, reinterpret_tensor(arg12_1, (2048, 2560)' in text
 result=dict(complete=True,CPU_only=True,quality_not_yet_established=True,model_executed=False,official_benchmark=False,adopted=False,rows_by_role=rows,block_by_M={'128':1024,'512':1024,'4096':512},expected_dequant_binaries=binaries,banks=first['banks'],packed_payload_bytes=sum(b['packed_payload_bytes'] for b in first['banks']),original_checkpoint_sha256=first['checkpoint_before'],holdouts={version:dict(directory=str(directory),gate_sha256=sha(directory/'INDEPENDENT_CPU_GATE_32g.json'),raw_result_sha256=gate['raw_result_sha256']) for version,(directory,gate,r) in gates.items()},compiled_BF16_lowering_evidence=dict(path=str(evidence),sha256=sha(evidence),scope='Existing generated M64 model fragment with nativeDown namespace lowers QKV/O linear to extern_kernels.mm. Exact published r2 execution linkage is NOT asserted.',selected_lines=[row.strip() for row in text.splitlines() if 'extern_kernels.mm' in row]));(BASE/'selected_plan.json').write_text(json.dumps(result,indent=2)+'\n');assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules);print(json.dumps({k:v for k,v in result.items() if k not in ('banks','holdouts','compiled_BF16_lowering_evidence')}))
if __name__=='__main__':main()
