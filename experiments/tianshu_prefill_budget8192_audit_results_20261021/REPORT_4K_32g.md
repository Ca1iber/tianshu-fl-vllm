# Budget8192 + qualified GUP prefill INT8: small4K +1.376%, new quality pending

BI-V15032G, original13serve argv,4K/1K,C64,N64,three rounds per arm,exclude first. Both arms keep the qualified NN GateUp decode, LM8BF16/LM64NTINT8 and GUP prefill INT8. Only actual default scheduling budget and corresponding startup shapes change; explicit user settings remain preserved. Unsupported M still delegates to BF16. No target sampling or checkpoint change.

| mode | round1 excluded | round2 | round3 | steady total tok/s | mean TTFT ms | spread |
|---|---:|---:|---:|---:|---:|---:|
|4096|5306.36708|5511.64881|5498.75991|5505.20436|10390.35784|0.23412%|
|8192|5365.70946|5576.50363|5585.46161|5580.98262|10275.37571|0.16051%|

Steady throughput +1.376484%; TTFT -1.106623%. All384timed requests succeeded. No new Triton autotune records in any round. These are small-screen numbers, not formalN256 or a replacement for published5586.28/Math102/105.

All445runtime sources and64observed environment keys are bound to both arms. Independent audit verifies actual worker and scheduler budgets4096/8192, native warm83/99, own numerically checked INT8 warm24/40, all42 actual layers at M4096/8192, Graph capture/replay, logical packed bank hashes, actual model memory profiling, zero4K preemptions and complete FutureKV release. Packed weights remain1,326,938,112bytes; actual KV blocks40723/40219. Report the504-block capacity reduction rather than assuming prior capacity persists. Candidate records show actual prefill INT8 M128/4096/5120/8192; these Python route counts do not count Graph replays or requests.

The frozen original controllers finish GPU workload but terminate audit_failed with KeyError('native_sha256'). Separate recovered_runtime_audit.py fixes only the actual metadata key native_source_sha256 and explicitly rebinds BASE to the frozen experiment. Entire runtime audit and independent pair verifier pass; raw phases/results, source files and CPU gate remain unchanged. Original GPU data was not rerun. Recovery/source hashes are retained. See RECOVERY_32g.md and CONTROL_RECOVERY_GATE_32g.json.

This gain exceeds both arms' spread and meets the planned threshold for fresh105-question Level3Math and originalN256 formal4K, with the same445source/64environment candidate profile. Quality and formal outcomes are still pending; adopted=false. Full16K is not routinely scheduled.

Raw responses/source snapshots remain at /workspace/benchmark-results/20261021_prefill_budget8192_service_{control,candidate}_tianshu. Only code, reports and key JSON are uploaded.
