# New QKV/O prefill quality — independently recounted

Fresh full Level3 C8 101/105; C64 102/105. Both strictly above95%. All105 per batch counted, including max-token finishes, with zeroAPIerrors/incomplete. See QUALITY_GATE_32g.json and KEY_SCORES_32g.csv for exact fractions, incorrect indices, stop reasons and immutable raw hashes. Full responses/predictions remain in the container.

Independent CPU verifies903model inputs (original890r2 exact subset),51controllers and every source/controller snapshot, actual13serveargv/63environment, original checkpoint SHA. New QKV/O and retained Down route/startup/profile structures independently rechecked and compared to original validator records. Other inherited attention/KV/scheduler/sampling/GU/LM checks passed in the original frozen validator; this CPU script does not separately rerun them.

C64 actually executed new nativeM4096 for all42QKV/all42O after ready. Both batches have zero new nativeM64 captures. Extra397135872bytes are in the real DeviceMemoryProfiler/profile path. C8 unhit paths are not claimed as evidence for C64. No prior scores reused and no original raw modified.

This is a model-quality gate, not a new throughput score or adoption. Next is the separately frozen short4K comparison against exact publishedr2; officialN256x4 only if that performance screen is meaningful and stable. No16K. Publishedr2 remains unchanged.
