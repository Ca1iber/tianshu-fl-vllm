# SPDX-License-Identifier: Apache-2.0
"""Only move standalone TopP's safe preselection threshold; exact mass fallback remains."""
import hashlib,importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'tianshu_topp_geometric_20261016/precision.py'
s=importlib.util.spec_from_file_location('_topp_preselection_geo_parent',PARENT);parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
NATIVE_SHA='64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
QUALIFIED_SHA='09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
MODES={'geo':0.,'shift_half':.5,'shift_one':1.}

def generate(raw,mode):
    assert hashlib.sha256(raw).hexdigest()==NATIVE_SHA and mode in MODES
    baseline=parent.render_source(raw,'geo');assert hashlib.sha256(baseline.encode()).hexdigest()==QUALIFIED_SHA
    start=baseline.index('        if TOPP_ENABLED and final_pivot == -float("inf"):');end=baseline.index('        # Sixth pass:',start)
    anchor='                outlier_pivot = avg_logit + std_logit * sigma\n'
    assert baseline[start:end].count(anchor)==1
    replacement=anchor if mode=='geo' else anchor.rstrip()+f' + {MODES[mode]} * std_logit * std_logit\n'
    part=baseline[start:end].replace(anchor,replacement);text=baseline[:start]+part+baseline[end:]
    assert text[:start]+part.replace(replacement,anchor)+text[start+len(part):]==baseline
    assert parent.topk_ast(text)==parent.topk_ast(baseline)
    return text,dict(mode=mode,mean_shift_factor=MODES[mode],native_sha256=NATIVE_SHA,qualified_parent_sha256=QUALIFIED_SHA,
        generated_sha256=hashlib.sha256(text.encode()).hexdigest(),only_standalone_preselection_threshold_changed=True,
        TopK_AST_unchanged=True,full_softmax_and_mass_fallback_unchanged=True,geometric_search_and_final_mask_unchanged=True,
        search_limit_unchanged=32,interval_tolerance_unchanged=1e-12,CPU_semantic_tolerance_unchanged=1e-6,p_and_RNG_unchanged=True)

def load_kernels(native,directory):
    raw=Path(native).read_bytes();directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);kernels={};proofs={}
    for mode in MODES:
        text,proof=generate(raw,mode);path=directory/f'generated_{mode}.py';path.write_text(text)
        name='_topp_preselect_'+mode+'_'+proof['generated_sha256'][:12];s=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(s);sys.modules[name]=module;s.loader.exec_module(module)
        kernels[mode]=module._topk_topp_kernel;proofs[mode]=dict(proof,generated_source=str(path))
    return kernels,proofs

def dependency_sources():
    return [Path(__file__),PARENT,parent.PARENT,parent.previous.PREVIOUS_PATH,parent.previous.previous.BASE_PATH,
            Path('/usr/local/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_triton.py')]
