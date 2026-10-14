# SPDX-License-Identifier: Apache-2.0
"""Change only native TopP tile and launch arguments, retaining its exact body."""
import hashlib
import json
from pathlib import Path

PLANS = [
    ("adopted", 8192, 4096, 16), ("identity", 8192, 4096, 16),
    ("b4096w8", 4096, 2048, 8), ("b4096w16", 4096, 2048, 16),
    ("b2048w8", 2048, 1024, 8), ("b16384w16", 16384, 8192, 16),
    ("b8192w32", 8192, 4096, 32), ("b16384w32", 16384, 8192, 32),
]


class Router:
    def __init__(self, kernel, plan, artifact_root=None):
        self.kernel, self.plan, self.metadata = kernel, plan, None
        self.artifact_root = artifact_root

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            label, tile, trunc, warps = self.plan
            assert kwargs["VOCAB_SIZE"] == 130560 and kwargs["BATCH_SIZE"] >= 8
            kwargs.update(BLOCK_SIZE=tile, BLOCK_SIZE_TRUNC=trunc, num_warps=warps, num_stages=1)
            compiled = self.kernel[grid](*args, **kwargs)
            self.metadata = dict(label=label, BLOCK_SIZE=tile, BLOCK_SIZE_TRUNC=trunc,
                num_warps=compiled.metadata.num_warps, num_stages=compiled.metadata.num_stages,
                shared=compiled.metadata.shared, grid=list(grid),
                n_regs=getattr(compiled, "n_regs", None), n_spills=getattr(compiled, "n_spills", None),
                VOCAB_SIZE=kwargs["VOCAB_SIZE"], TOPK_ENABLED=kwargs["TOPK_ENABLED"], TOPP_ENABLED=kwargs["TOPP_ENABLED"])
            assert self.metadata["num_warps"] == warps and self.metadata["num_stages"] == 1
            if self.artifact_root is not None and kwargs["BATCH_SIZE"] == 64 and not kwargs["TOPK_ENABLED"]:
                root = Path(self.artifact_root) / label
                if not root.exists():
                    root.mkdir(parents=True)
                    files = {}
                    for name, value in compiled.asm.items():
                        assert name.replace("_", "").isalnum()
                        raw = value.encode() if isinstance(value, str) else value if isinstance(value, bytes) else None
                        if raw is not None:
                            (root / name).write_bytes(raw)
                            files[name] = hashlib.sha256(raw).hexdigest()
                    (root / "manifest.json").write_text(json.dumps(dict(metadata=self.metadata, files=files), indent=2) + "\n")
            return compiled
        return launch
