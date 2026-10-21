# SPDX-License-Identifier: Apache-2.0
"""Process-local seven-slot DSpark input limit; other drafts retain native rules."""
from functools import wraps

ARCHITECTURES = frozenset(('Qwen3DSparkModel', 'DFlashQwen3DSparkModel'))
QUERY_SLOTS = 7

def is_dspark(spec):
    if spec is None or not spec.use_dflash():
        return False
    draft = getattr(spec, 'draft_model_config', None)
    hf = getattr(draft, 'hf_config', None)
    return bool(ARCHITECTURES.intersection(getattr(hf, 'architectures', ()) or ()))

def seven_slot_input_fit(original):
    @wraps(original)
    def fits(self, common_attn_metadata):
        if common_attn_metadata is None or not is_dspark(self.speculative_config):
            return original(self, common_attn_metadata)
        return (common_attn_metadata.max_seq_len + QUERY_SLOTS
                <= self.effective_drafter_max_model_len)
    return fits

def install(module):
    cls = module.ModelRunnerFL
    assert not getattr(cls._input_fits_in_drafter, '_tianshu_dspark_seven_slot', False)
    wrapped = seven_slot_input_fit(cls._input_fits_in_drafter)
    wrapped._tianshu_dspark_seven_slot = True
    cls._input_fits_in_drafter = wrapped
