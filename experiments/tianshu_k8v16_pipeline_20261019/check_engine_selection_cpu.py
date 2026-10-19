# SPDX-License-Identifier: Apache-2.0
"""Exercise actual Worker loader selection without importing or launching a GPU engine."""
import ast
import functools
import importlib.util
from pathlib import Path
from types import SimpleNamespace

base = Path(__file__).resolve().parent
node = next(n for n in ast.parse((base / 'hooks/sitecustomize.py').read_text()).body
    if isinstance(n, ast.ClassDef) and n.name == 'Loader')
for mode in ('control', 'candidate'):
    calls = []
    class Original:
        def exec_module(self, module): calls.append('load_native_worker')
    class Worker:
        def __init__(self): calls.append('native_init')
    def make_spec(name, path):
        calls.append(str(path))
        def execute(module):
            module.install = lambda worker: calls.append('install_selected')
        return SimpleNamespace(name=name, path=path, loader=SimpleNamespace(exec_module=execute))
    fake_import = SimpleNamespace(abc=SimpleNamespace(Loader=object), util=SimpleNamespace(
        spec_from_file_location=make_spec, module_from_spec=lambda spec: SimpleNamespace()))
    environment = dict(BASE=base, functools=functools, Path=Path, importlib=fake_import,
        os=SimpleNamespace(environ={'TIANSHU_K8V16_SERVICE_MODE': mode}),
        sys=SimpleNamespace(path=[], modules={}))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(base / 'hooks/sitecustomize.py'), 'exec'), environment)
    module = SimpleNamespace(WorkerFL=Worker)
    environment['Loader'](Original()).exec_module(module)
    module.WorkerFL()
    target = base if mode == 'candidate' else base.parent / 'tianshu_soft8_engine_v2_20261019'
    assert calls == ['load_native_worker', 'native_init', str(target / 'engine_hooks.py'), 'install_selected']
    assert environment['sys'].path == [str(target)]
print(dict(complete=True, CPU_only=True, actual_Worker_loader_modes=2,
    baseline_uses_qualified_byte_engine=True, candidate_uses_new_K8V16_engine=True))
