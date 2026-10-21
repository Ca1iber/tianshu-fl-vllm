# SPDX-License-Identifier: Apache-2.0
"""Wrap the existing qualified kernel route; preserve the real sampler and aliases."""
import hashlib
import importlib.util
from pathlib import Path

NATIVE_SHA = '64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
GEO_SHA = '09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'


def supported(args, kwargs):
    if not (len(args) >= 3 and kwargs.get('TOPP_ENABLED') and not kwargs.get('TOPK_ENABLED')
            and 8 <= kwargs.get('BATCH_SIZE', 0) <= 64
            and kwargs.get('VOCAB_SIZE') == 130560
            and kwargs.get('BLOCK_SIZE') == 8192 and kwargs.get('BLOCK_SIZE_TRUNC') == 4096):
        return False
    logits = args[0]
    return (logits.ndim == 2 and tuple(logits.shape) == (kwargs['BATCH_SIZE'],130560)
            and str(logits.dtype) == 'torch.float32' and logits.device.type == 'cuda'
            and logits.stride(1) == 1)


def install(native, geo, enabled, observer=None, artifact_root=None):
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == NATIVE_SHA
    assert hashlib.sha256(Path(geo.__file__).read_bytes()).hexdigest() == GEO_SHA
    assert not hasattr(native, '_tianshu_grid_route')
    source = Path(__file__).resolve().with_name('kernel_route.py')
    spec = importlib.util.spec_from_file_location('_tianshu_grid_measured_router', source)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    original = native._topk_topp_kernel
    inner = helper.Router(original, ('fixed32_grid2',32768,16384,16,2),
                          artifact_root=artifact_root, corrected_kernel={'geo':geo._topk_topp_kernel})

    class Route:
        def __init__(self):
            self.enabled = bool(enabled)
            self.last_metadata = None

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                active = self.enabled and supported(args,kwargs)
                result = inner[grid](*args,**kwargs) if active else original[grid](*args,**kwargs)
                self.last_metadata = dict(active=active, enabled=self.enabled,
                    batch=kwargs['BATCH_SIZE'], topk=kwargs['TOPK_ENABLED'], topp=kwargs['TOPP_ENABLED'],
                    input_grid=list(grid), actual_candidate=inner.metadata if active else None,
                    input_row_stride=args[0].stride(0),input_col_stride=args[0].stride(1),
                    GEO_source_sha256=GEO_SHA, sampler_RNG_untouched=True,
                    compiled_metadata={k:getattr(result.metadata,k,None) for k in ('num_warps','num_stages','shared')})
                if observer is not None:
                    observer(self.last_metadata)
                return result
            return launch

    route = Route()
    native._topk_topp_kernel = route
    native._tianshu_grid_route = route
    return route
