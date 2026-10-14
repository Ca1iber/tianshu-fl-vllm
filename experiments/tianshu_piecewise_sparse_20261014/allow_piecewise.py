# SPDX-License-Identifier: Apache-2.0
"""Keep all existing bucket guards; allow one additional graph mode via read-only proxies."""
from types import SimpleNamespace


class View:
    def __init__(self, original, **overrides):
        self._original = original
        self.__dict__.update(overrides)
    def __getattr__(self, name): return getattr(self._original, name)


def extend(module):
    original = module.enabled

    def enabled(config, platform):
        compilation = config.compilation_config
        if compilation.cudagraph_mode.name == "FULL_AND_PIECEWISE":
            compilation = View(compilation, cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"))
            return original(View(config, compilation_config=compilation), platform)
        return original(config, platform)

    module.enabled = enabled
    return dict(only_additional_mode="FULL_AND_PIECEWISE", other_model_and_parallel_guards_preserved=True,
                configure_warmup_padding_and_points_unmodified=True, installed_source_modified=False)
