# SPDX-License-Identifier: Apache-2.0
"""Freeze current paired-K source and T16; vary only warp/stages at n>24."""
import hashlib,importlib.util,sys
from pathlib import Path
PARENT_SHA='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
CONFIGS={'current':(1,1),'w2_s1':(2,1),'w4_s1':(4,1),'w1_s2':(1,2),'w2_s2':(2,2),'w1_s3':(1,3),'w2_s3':(2,3)}
MODES=tuple(CONFIGS)
SHORT_SPLITS={mode:4 for mode in MODES}
def configuration(mode,n):
    assert mode in MODES and 1<=n<=64
    warps,stages=CONFIGS[mode] if n>24 else CONFIGS['current']
    return dict(TILE_SIZE=16,num_warps=warps,num_stages=stages)
def configured_segments(mode,n,parent_segments):
    assert mode in MODES and n>0
    return parent_segments

def render(parent,mode):
    parent=Path(parent);text=parent.read_text()
    assert hashlib.sha256(text.encode()).hexdigest()==PARENT_SHA and mode in MODES
    return text,dict(mode=mode,parent_sha256=PARENT_SHA,generated_sha256=PARENT_SHA,
        original_reader_source_bitwise_unchanged=True,changed_functions=[],function_ABI_unchanged=True,
        only_external_warps_and_stages_changed=True,TILE_SIZE_always16=True,eligible_batches='num_seqs>24',
        short_active_segments=4,segment_capacity_and_KV_writer_and_format_unchanged=True,
        selected_configuration_for_B64=configuration(mode,64),Math_or_service_not_claimed=True)
def load(parent,directory,mode):
    text,proof=render(parent,mode);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    path=directory/f'generated_{mode}.py';path.write_text(text)
    name='_pair_T16_pipeline_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module,dict(proof,generated_source=str(path))
