# SPDX-License-Identifier: Apache-2.0
"""Identical qualified paired-K source, only component launch tile/warp differs."""
import hashlib,importlib.util,sys
from pathlib import Path
PARENT_SHA='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
MODES=('current','tile32_w1','tile32_w2','tile64_w2')
SHORT_SPLITS={mode:4 for mode in MODES}
CONFIGS={'current':(16,1),'tile32_w1':(32,1),'tile32_w2':(32,2),'tile64_w2':(64,2)}
def configuration(mode,n):
    assert mode in MODES and 1<=n<=64
    tile,warps=CONFIGS[mode] if n>24 else CONFIGS['current']
    return dict(TILE_SIZE=tile,num_warps=warps,num_stages=1)
def configured_segments(mode,n,parent_segments):
    assert mode in MODES and n>0
    return parent_segments

def render(parent,mode):
    parent=Path(parent);text=parent.read_text()
    assert hashlib.sha256(text.encode()).hexdigest()==PARENT_SHA and mode in MODES
    return text,dict(mode=mode,parent_sha256=PARENT_SHA,generated_sha256=PARENT_SHA,
        original_reader_source_bitwise_unchanged=True,changed_functions=[],function_ABI_unchanged=True,
        only_external_launch_tile_and_warps_changed=True,eligible_batches='num_seqs>24',
        short_active_segments=4,segment_capacity_and_KV_writer_and_format_unchanged=True,
        selected_configuration_for_B64=configuration(mode,64),Math_or_service_not_claimed=True)
def load(parent,directory,mode):
    text,proof=render(parent,mode);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    path=directory/f'generated_{mode}.py';path.write_text(text)
    name='_pair_tile_sweep_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module,dict(proof,generated_source=str(path))
