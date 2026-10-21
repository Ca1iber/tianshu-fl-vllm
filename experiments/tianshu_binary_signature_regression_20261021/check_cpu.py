# SPDX-License-Identifier: Apache-2.0
"""Mutation tests for the exact non-debug ELF gate, using real cached code."""
import base64,hashlib,importlib.util,json,struct,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;helper=BASE.parent/'tianshu_k8v16_bm8_abi_recovery_20261021/binary_signature.py'
s=importlib.util.spec_from_file_location('_binary_gate_mutation_test',helper);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
key='18a1d2675e864b3831a7837ab787ea3d73af1a00188bcde23657c8144d5866ff';folder=base64.b32encode(bytes.fromhex(key)).decode().rstrip('=');path=Path('/root/.triton/cache')/folder/'kernel_unified_attention.cubin';original=path.read_bytes();baseline=m.signature(original)
offset=struct.unpack_from('<Q',original,40)[0];entry,count,strings=struct.unpack_from('<HHH',original,58);rows=[struct.unpack_from('<IIQQQQIIQQ',original,offset+i*entry) for i in range(count)];names=original[rows[strings][4]:rows[strings][4]+rows[strings][5]];lookup={names[r[0]:].split(b'\0',1)[0].decode():(i,r) for i,r in enumerate(rows)}
checks=[]
for name in ('.text','.rodata','.note','.symtab'):
 _,r=lookup[name];changed=bytearray(original);changed[r[4]]^=1;assert m.signature(bytes(changed))['non_debug_signature_sha256']!=baseline['non_debug_signature_sha256'];checks.append(dict(case='mutate_'+name,rejected_by_code_signature=True))
i,r=lookup['.debug_info'];changed=bytearray(original);changed[r[4]]^=1;observed=m.signature(bytes(changed));assert observed['non_debug_signature_sha256']==baseline['non_debug_signature_sha256'] and observed['full_binary_sha256']!=baseline['full_binary_sha256'];checks.append(dict(case='mutate_nonloaded_debug_info_only',accepted_with_full_SHA_difference_disclosed=True))
changed=bytearray(original);struct.pack_into('<Q',changed,offset+i*entry+8,2)
try:m.signature(bytes(changed))
except AssertionError:checks.append(dict(case='debug_info_marked_loadable',rejected=True))
else:raise AssertionError('loadable debug section accepted')
changed=bytearray(original);changed[18]^=1;assert m.signature(bytes(changed))['non_debug_signature_sha256']!=baseline['non_debug_signature_sha256'];checks.append(dict(case='mutate_ELF_machine',rejected_by_code_signature=True))
try:m.signature(original[:64])
except AssertionError:checks.append(dict(case='truncated_ELF',rejected=True))
else:raise AssertionError('truncated ELF accepted')
assert len(checks)==8 and 'torch' not in sys.modules
report=dict(complete=True,CPU_only=True,Torch_and_GPU_libraries_not_imported=True,actual_cached_binary_used=True,actual_cached_binary_sha256=hashlib.sha256(original).hexdigest(),helper_source=str(helper),helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),mutation_cases=checks,scope='Loaded code/data/metadata/symbol and machine changes rejected; only flags0 nonloaded debug_info accepted with full SHA retained')
assert not (BASE/'CPU_GATE_32g.json').exists();(BASE/'CPU_GATE_32g.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
