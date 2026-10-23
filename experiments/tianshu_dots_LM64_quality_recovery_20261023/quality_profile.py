# SPDX-License-Identifier: Apache-2.0
"""CPU-only explicit guard reconciliation; original GPU/controller records stay failed."""
import importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
original=load('_LM64_recovery_original_quality',EXP/'tianshu_dotsaddz_LM64_quality_20261023/quality_profile.py');adapted=load('_LM64_recovery_explicit_Down_precision_guard',BASE/'adapted_down_audit.py');original.down.own=adapted
audit_runtime=original.audit_runtime;audit_math=original.audit_math;digest,read,frozen_check=original.digest,original.read,original.frozen_check
