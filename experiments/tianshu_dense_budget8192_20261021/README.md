# Current W8A8 profile: scheduler budget4096 versus8192

This is a fresh screen on the qualified5482.79 profile, with original13-item serve argv, seq64, async scheduling, future-KV reservation, paired-K reader, integer GateUp M8/M64, integer LM M64, BF16 small LM and BF16 prefill linear unchanged. It does not use the experimental MLP fusion boundary.

The outer EngineArgs wrapper runs after the qualified budget4096 and seq64 selectors. Its pre-native guard is the same installed vendor/model/shape/API/parallel/default guard. Only eligible absent budget defaults are changed; explicit values are preserved. Inner default records describe their intermediate4096 state; actual worker/scheduler and an additional outer record must prove final4096/8192. Runtime warmup must complete83/99 points through actual budget, respectively. No installed source is edited.

CPU check executes the installed native EngineArgs AST with both actual inherited wrappers, checks13 guard cases, compares all owner state against explicit budget4096/8192 plusseq64, verifies exception identity/rollback, and composes both complete profiles. Both arms screen4K/1K C64 N64,3 rounds, first excluded. Need >=0.5% paired and prior-qualified same-client gain, TTFT no worse than1%, spread<1%, and no postfirst autotune before fresh Math and formalN256. The old10/19 BF16-profile8192 result+1.06% is history, not evidence for the new integer model.

Code/raw remain in the Tianshu container; GitHub stores code/report/key JSON. No new Math or16K has run. All timed requests,43 packed weight signatures, complete source snapshots, actual environments, KV allocation, real capture/replay and precision routes are independently audited.
