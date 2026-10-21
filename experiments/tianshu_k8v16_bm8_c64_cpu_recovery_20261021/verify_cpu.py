# SPDX-License-Identifier: Apache-2.0
"""CPU-only reconciliation of completed Math105 after nonnumerical CSV EOL change."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import traceback

BASE = Path(__file__).resolve().parent
RAW = Path('/workspace/benchmark-results/20261021_BM8_v3_c64_quality_tianshu')
OUT = Path('/workspace/benchmark-results/20261021_BM8_v3_C64_CPU_recovery_tianshu')
VALIDATION = BASE.parent / 'tianshu_k8v16_bm8_validation_v3_20261021'
SCREEN = BASE.parent / 'tianshu_k8v16_bm8_service_results_v3_20261021'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

class NoGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'vllm', 'flag_gems'}:
            raise ImportError('GPU framework import prohibited in CPU reconciliation: ' + fullname)

sys.meta_path.insert(0, NoGPUImports())
assert not OUT.exists()
OUT.mkdir()
original_hashes = {str(RAW / name): sha(RAW / name) for name in ('state.json', 'result.json')}
gate = dict(complete=False, CPU_only=True, GPU_framework_imports_prohibited=True,
            original_controller_passed=False, original_state_preserved=True,
            new_GPU_requests=0, official_C8_Math_not_run=True, formal_N256_not_run=True,
            adopted=False, raw_root=str(RAW), recovery_raw_root=str(OUT),
            verifier_sha256=sha(__file__), original_raw_hashes=original_hashes)
try:
    state = read(RAW / 'result.json')
    assert state['phase'] == 'failed' and state['error'] == 'AssertionError()'
    assert state['evalscope_exit_code'] == 0
    assert (RAW / 'after_metrics.txt').is_file()
    assert len(state['frozen_sources']) == 500 and len(state['controller_sources']) == 14
    for mapping in ('frozen_sources', 'controller_sources'):
        for path, expected in state[mapping].items():
            assert sha(path) == expected, (mapping, path)
    csv = SCREEN / 'KEY_ROUNDS_32g.csv'
    snapshot = RAW / 'controller_source_snapshot' / csv.relative_to('/')
    lf = RAW / 'KEY_ROUNDS_LF_at_failed_postcheck.csv'
    assert csv.read_bytes() == snapshot.read_bytes()
    assert csv.read_bytes().replace(b'\r\n', b'\n') == lf.read_bytes()
    assert csv.read_bytes() != lf.read_bytes()
    screen = read(SCREEN / 'SCREEN_GATE_32g.json')
    assert state['frozen_sources'] == screen['candidate_frozen_sources']
    gate.update(original500_numerical_sources_unchanged=True, all14_original_controller_hashes_restored=True,
                only_changed_dependency_was_nonnumerical_CSV_line_endings=True,
                restored_CSV_sha256=sha(csv), failed_postcheck_LF_CSV_sha256=sha(lf),
                matched_screen_gate_sha256=sha(SCREEN / 'SCREEN_GATE_32g.json'))
    sys.path.insert(0, str(VALIDATION))
    from auxiliary_profile import audit_runtime, audit_math
    profile = audit_runtime(RAW, state, official=False)
    quality = audit_math(RAW, state, profile, write=False)
    assert quality['total'] == 105 and quality['correct'] == 103
    assert quality['BM8_actual_C64_replay_verified'] and quality['official_Math8_not_replaced']
    assert not quality['fresh_math_for_original_command_profile']
    assert quality['model_frozen_sources'] == state['frozen_sources']
    (OUT / 'profile_audit.json').write_text(json.dumps(profile, indent=2) + '\n')
    (OUT / 'accuracy_audit.json').write_text(json.dumps(quality, indent=2) + '\n')
    gate.update(complete=True, supplementary_C64_quality=True, correct=quality['correct'],
                total=105, accuracy=quality['accuracy'], accuracy_requirement_passed=quality['official_requirement_passed'],
                BM8_actual_C64_replay_verified=True, actual_M4096_range_fused_all42_verified=True,
                all105_predictions_and_reviews_independently_checked=True,
                actual_42_GateUp_and_M64_LM_INT8_verified=True,
                original_frozen_audit_functions_reexecuted_on_CPU=True,
                original_failed_controller_not_rewritten_as_success=True,
                model_frozen_sources=state['frozen_sources'],
                raw_accuracy_audit_sha256=sha(OUT / 'accuracy_audit.json'),
                raw_profile_audit_sha256=sha(OUT / 'profile_audit.json'),
                prediction_and_review_hashes=quality['raw_files'])
except Exception:
    gate['error'] = traceback.format_exc()
finally:
    assert all(sha(path) == expected for path, expected in original_hashes.items())
    assert not any(name.split('.')[0] in {'torch', 'triton', 'vllm', 'flag_gems'} for name in sys.modules)
    (BASE / 'C64_CPU_RECOVERY_GATE_32g.json').write_text(json.dumps(gate, indent=2) + '\n')
    print(json.dumps({k: gate.get(k) for k in ('complete', 'correct', 'total', 'accuracy', 'error')}, indent=2))
if not gate['complete']:
    raise SystemExit(1)
