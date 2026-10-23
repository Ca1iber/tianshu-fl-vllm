# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent

def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
original=load('_short_recovery_original_all_checks',BASE.parent/'tianshu_W4_LMpad_short4k_v3_20261023/audit_profile.py')
adapted=load('_short_recovery_LM8_BF16_to_actual_W4',BASE/'adapted_prefill_audit.py');adapted.parent=original.dense
original.candidate.legacy.legacy.lower=adapted
original.candidate.legacy.legacy.prefill.parent.parent=adapted
audit_startup,audit_runtime=original.audit_startup,original.audit_runtime
