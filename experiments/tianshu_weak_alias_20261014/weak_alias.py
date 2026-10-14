# SPDX-License-Identifier: Apache-2.0
"""Non-owning tensor alias for graph-pool managed outputs, with explicit source identity."""
import hashlib
import os
from pathlib import Path


def load():
    from torch.utils.cpp_extension import load_inline
    source = Path(__file__).with_suffix(".cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = Path("/tmp") / ("tianshu_weak_alias_build_" + digest[:12])
    directory.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("MAX_JOBS")
    os.environ["MAX_JOBS"] = "2"
    try:
        module = load_inline(name="_tianshu_weak_alias_" + digest[:12],
                             cpp_sources=source.read_text(), functions=["weak_alias"],
                             extra_cflags=["-O2"], with_cuda=False, build_directory=str(directory), verbose=False)
    finally:
        if previous is None: os.environ.pop("MAX_JOBS", None)
        else: os.environ["MAX_JOBS"] = previous
    return module, dict(cpp_source_sha256=digest, shared_object=module.__file__,
                        shared_object_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                        build_directory=str(directory), graph_pool_must_own_underlying_allocation=True)
