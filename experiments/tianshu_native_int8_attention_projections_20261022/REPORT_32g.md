# Native INT8 QKV/O components — promising prefill, no model claim

Eight role/row cases,24 phases,4 native variants per case completed full INT32 and sequential-scale BF16 equality versus the separate Triton NT diagnostic. Independent CPU recomputed full activation packing/static original weights and selected INT64 outputs/BF16 projections/FP64 BF16 control bounds. Sources, snapshots, checkpoint, graph-poison/eager, input/weights/RNG/canaries passed.

Explicit vendor BF16 control vs fastest graph candidate (ms):

|Role|M|BF16|INT8|inverse-time gain|
|---|---:|---:|---:|---:|
|QKV|64|0.03247923|0.02534615|28.14%|
|QKV|128|0.04936692|0.02938000|68.03%|
|QKV|512|0.10606693|0.05349000|98.29%|
|QKV|4096|0.49421811|0.35554502|39.00%|
|O|64|0.02370423|slower|negative|
|O|128|0.03358731|0.02820423|19.09%|
|O|512|0.08099347|0.04226038|91.65%|
|O|4096|0.38852999|0.29719615|30.73%|

QKV relative L2 versus original BF16~2.30%; O~1.30%. These are component errors, not model quality. Actual untouched QKV/O runtime route has not yet been verified; explicit vendor timing alone does not establish model gain. An independent all42-layer holdout selects one NN bank representation, M128/512/4096 only, block1024/1024/512. Small decode excluded; full C8/C64 Math>95% still required before promotion. R2 remains unchanged.
