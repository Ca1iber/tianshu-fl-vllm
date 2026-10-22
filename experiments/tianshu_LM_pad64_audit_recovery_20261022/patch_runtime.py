# SPDX-License-Identifier: Apache-2.0
import importlib.util,sys,types
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
TARGETS={'math':(EXP/'tianshu_nn_silu_fx_c64_quality_20261021/dense_quality_audit.py',BASE/'math_dense_audit.py'),'official':(EXP/'tianshu_nn_silu_fx_screen_20261021/dense_audit.py',BASE/'official_dense_audit.py')}
def modules(root):
 queue=[root];seen=set()
 while queue:
  m=queue.pop()
  if id(m) in seen:continue
  seen.add(id(m));p=getattr(m,'__file__',None)
  if not p or not Path(p).is_relative_to(EXP):continue
  queue.extend(v for v in vars(m).values() if isinstance(v,types.ModuleType));yield m,Path(p).resolve()
def origins(root,kind):return [str(Path(m.audit_runtime.__code__.co_filename).resolve()) for m,p in modules(root) if p==TARGETS[kind][0].resolve()]
def install(root,kind):
 target,newfile=TARGETS[kind];patched=[]
 for m,p in list(modules(root)):
  if p!=target.resolve():continue
  name=f'_LM_correct_actual_{kind}_{len(patched)}';s=importlib.util.spec_from_file_location(name,newfile);replacement=importlib.util.module_from_spec(s);sys.modules[name]=replacement;s.loader.exec_module(replacement);replacement.parent=m.parent;m.audit_runtime=replacement.audit_runtime;patched.append(str(target))
 assert patched and all(p==str(newfile.resolve()) for p in origins(root,kind));return patched
