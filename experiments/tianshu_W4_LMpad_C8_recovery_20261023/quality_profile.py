# SPDX-License-Identifier: Apache-2.0
import importlib.util,sys,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
original=load('_LMpad_C8_recovery_original',BASE.parent/'tianshu_W4_LMpad_quality_v2_20261023/quality_profile.py')
adapted=load('_LMpad_C8_recovery_actual_NN8',BASE/'adapted_C8_dense_audit.py')
adapted.parent=original.original_C8_parent;original.down.core.q8.parent=adapted
audit_runtime,audit_math,digest,read,frozen_check=original.audit_runtime,original.audit_math,original.digest,original.read,original.frozen_check

def verify_freeze():subprocess.run(['sha256sum','-c',str(BASE/'EXECUTION.sha256')],cwd=BASE.parent.parent,check=True,stdout=subprocess.DEVNULL)
