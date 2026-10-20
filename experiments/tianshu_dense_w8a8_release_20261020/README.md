# Selective W8A8 original-command profile

Prepared from the actual fresh MATH runtime profile: 311 runtime sources, 9 controllers, 57 selected environment variables, original 13 serve argv. This preparation is not an adoption decision; independent new quality and formal 4K results determine that separately. The small N64 screen is +2.5786%; it is not the formal N256 score.

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_dense_w8a8_release_20261020/restore_generated_sources.sh --check
bash experiments/tianshu_dense_w8a8_release_20261020/serve_tianshu_minicpm.sh
```

The direct Bash launcher checks every runtime/controller SHA before exec and clears inherited profile flags, including VLLM_DISABLE_COMPILE_CACHE, then exports the exact measured environment with relocated record paths. VLLM_DISABLE_COMPILE_CACHE=1 is necessary: older AOT graphs can bypass new GateUp hooks. Original model checkpoint is unchanged. Loader-time extra nonpersistent INT8/scale banks are regenerated from it, within the actual FL memory-profiler context. Model files, packed weight payloads, full responses, compiler caches, and raw benchmark tensors are not in this bundle. Only 26 small generated Python sources are included for reproducibility; existing mismatched files are never overwritten by restore.

CPU-only dry run:
```bash
TIANSHU_DENSE_W8A8_PROFILE_DRY_RUN=1 bash experiments/tianshu_dense_w8a8_release_20261020/serve_tianshu_minicpm.sh
```

The launcher itself is not a separate GPU benchmark; actual measured server argv is executed directly. No full 16K result exists for this new profile; the previous full-16K qualified result uses the earlier future-KV profile.
