# Next KV/batching direction: stage using prefill lengths, then evaluate arena capacity

Research evidence only, not implemented or a measured performance improvement. Current budget8192 candidate is still in fresh Math; do not alter it or its445runtime/16primary sources. Staticcompile420-source experiment is CPU-prepared separately and must wait for the current quality/formal queue to finish.

## Concrete staging issue

Current tianshu_k8v16_pipeline_20261019/routing.py copies q/out/cu/seq_lengths/block_table to the prefill tail, but leaves max_seqlen_k at the whole-batch maximum. BoundedDensePrefill.shape_for then allocates logical dense rows and launches n_prefill*ceil(global_max_K/16) pages, including inactive pages attributable to an unrelated long decode request. Actual gather masks inactive cells, so this is overhead/capacity usage rather than proof of wrong output. Existing mixed splitting itself is already active and must not be reimplemented.

CommonAttentionMetadata.seq_lens_cpu_upper_bound is explicitly documented precise for prefill rows, including async-spec's prefill rows. Actual ModelRunnerFL constructs it from optimistic_seq_lens_cpu and keeps it even when deprecated _seq_lens_cpu is cleared in async spec. Currentprofile speculative_config=None. The deprecated seq_lens_cpu property can trigger GPU->CPU copy if absent; do not use that property or add a synchronization.

Possible independent implementation: wrap hybrid.route_from_common, preserve the original Route and GPU cu subtraction, attach max(seq_lens_cpu_upper_bound[num_decodes:num_reqs]) once per step after validating CPU device/type/shape/positive lengths/full key length>=query length and <=global bound. Missing/invalid metadata falls back to the original bound. At BoundedDensePrefill.__call__ only, read the same Route via hybrid._ROUTE ContextVar and pass a copied prefill kwargs with the tighter max_seqlen_k. The existing stage return carries it to FA2. Do not change the original whole-batch kwargs before splitting: that would also change decode segmentation based on unrelated prefill lengths. No input/KV/writer/causal/sampling change. Do not replace full prefill key length with query length when prefix context exists.

CPU tests must cover pure/mixed batches, prefix context, padded/invalid metadata and unavailable CPU values; GPU component must compare every active gathered value/guard between old and tighter bounds. Keep arena8192 first to isolate the bound-only effect; actual runtime backend code and C8/C64Graph paths must remain verified. Source review alone does not establish end-to-end gain or universal numerical equivalence.

## Potential memory threshold, not an admission result

The observed8192 candidate has40219KVblocks including one null block. FutureKV forecasts ceil((prompt+max_tokens)/16), so nominal independent16K/1K demands1088blocks per request. Ignoring allocator/prefix/liveness details,40218 usable blocks fit36forecasts;37need40256usable. Need38additional total blocks. Existing one shared BF16 arena8192pages occupies128MiB plus guard. Reducing to6144pages would nominally save32MiB; each GPUKVblock across42layers is516096bytes, suggesting65moreblocks and nominal37forecasts. This is CPU arithmetic, not proof of actual memory recovery or37live requests. Actual activation peaks, allocator rounding, retained allocations, prefix reuse and native reserves can alter it.

A separate capacity experiment can exploit the safe existing bounded fallback and target a long-context bottleneck. Keep it independent of the bound-only experiment. Use actual per-profile KV capacity/admissions and no full16K until a justified diagnostic/screen indicates value. Do not infer a16Kscore from the4Kcandidate gain. Current partialMath records show max observed stage pages in the attached observation JSON; that is not a universal bound or a replacement for newMath validation after a profile change.

## Source anchors checked in the live container

- hybrid route: vllm_fl/worker/iluvatar_hybrid_attention.py SHA2565c55a8d57e438b36d176555fd9273ab9ba665dada4c57d57c19633394c2ed4e4.
- CommonAttentionMetadata CPU contract: /usr/local/lib/python3.12/site-packages/vllm/v1/attention/backend.py SHA256c524988a2223ec556bdb128a31c255075f0d07e7643e803b447975cd9c30a6b8.
- actual metadata construction: vllm_fl/worker/model_runner.py SHA256b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e.
- future reservation: experiments/tianshu_future_kv_reservation_20261020/policy.py, actual forecast/admission code inspected.
- null block: /usr/local/lib/python3.12/site-packages/vllm/v1/core/block_pool.py, takes one block out of the free queue and computes total_gpu_blocks=num_gpu_blocks-1.
