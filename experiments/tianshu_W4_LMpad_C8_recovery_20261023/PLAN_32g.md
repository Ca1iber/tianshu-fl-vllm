# C8 CPU audit reconciliation, 2026-10-23

Original 944 model sources and 97 GPU controller sources remain unchanged. C8 completed all 105 requests once; original result remains audit_failed, AssertionError((8, 0)). No GPU rerun or score filtering is allowed.

The inherited C8 dense guard asks for the original NT8 warm-capture record. The final Graph8 actually uses NN8 and has 42 captured banks linked to 39138 replays. Replace only that stale capture selector by the actual NN8 evidence, requiring all 42 exact prefixes, frozen NN/NT/policy hashes, selected NN configuration, real Graph key, capture, replay and frame context. Preserve original NT checks and all downstream NN packing, precision, memory, KV, attention, scheduler, sampling and scoring checks.

Recovery runs CPU only after freezing these sources. Verify all 944 original runtime and 97 original controller current and snapshot hashes, all105 raw predictions/reviews/report and original checkpoint. Output a separate CPU_recovery_complete result with original_experiment_passed=false and GPU_rerun=false. C64 remains pending and is necessary before performance testing. Never rewrite original result or raw evidence.

Initial CPU attempt stopped before runtime audit because it assumed 95 controllers from the parent; actual v2 original GPU snapshot has 97. Preserved initial freeze and failure log; corrected the exact count to97 and froze again. No GPU or scoring rerun occurred.
