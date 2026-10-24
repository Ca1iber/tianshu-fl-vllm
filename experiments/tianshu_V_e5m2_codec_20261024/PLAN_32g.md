# V-only E5M2 storage experiment

Keep the adopted K software E4M3 bytes, Q/K score computation, softmax, page addressing, launch policy and reducer. Candidate V byte codes are the high byte of IEEE FP16, using nearest-even rounding and saturation to ±57344; NaNs become signed zero. The writer never emits the eight reserved Inf/NaN codes. This is runtime KV compression, not a model checkpoint change or native FP8 hardware claim.

Two separately measured readers share identical encoded V bytes: BF16_PV decodes back to BF16, retaining the current probability cast; FP16_PV returns FP16 and changes the probability cast for P×V to FP16, while Q/K remain BF16. Report this precision change explicitly. A byte page shrinks from 12288 to 8192 bytes; this does not prove an end-to-end speedup.

CPU prerequisite: exhaustively compare all 65536 BF16 encodings against an independent real-number rounding reference, decode all 256 codes, and prove reversibility of the generated source changes. No Torch/Triton import or GPU access in the CPU check.

Only after the active joint-QKV/O paired screen has terminated and GPU is actually idle: verify the GPU writer on all BF16 bit patterns, finite decoder values/signs and reserved-code classification. Then compare current K8/V16 against both candidates with the exact qualified S16 reader/launch policy, writer+decode Graph timings, changed-data replay, full GPU FP32 and selected CPU references, cache/storage/output/scratch/RNG guards, and alternating trial order. Require meaningful B64 4K and 5K reductions (at least 5%) before considering engine integration. Preserve negative results. Any engine integration requires fresh C64 and C8 Math >95% and fresh-input paired 4K service evidence before the original full benchmark. No automatic adoption or 16K benchmark.

All work and raw results remain in the container; GitHub receives code, reports and key results. Hard GPU stop 2026-10-24 23:58 BJT.
