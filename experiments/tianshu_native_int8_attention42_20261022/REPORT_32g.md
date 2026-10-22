# All42 QKV/O prefill native INT8 holdout

All84 original banks,504 phase artifacts and129024 selected INT64/quantized BF16/original FP64 output cells independently verified. Full GPU native INT32/BF16 equals separate Triton NT diagnostic; full84-bank graph poison/eager hashes, all weight/source/snapshot/checkpoint/input/RNG/canaries passed.

Explicit vendor mm BF16 vs NN native INT8 combined-pair normalized per layer: M1280.08801923->0.06282826ms (+40.09%), M5120.19886868->0.09742006ms (+104.14%), M40960.87791467->0.64573246ms (+35.96%). QKV/O consume the same independent activation matrix, so these are projection memory-traffic components, not attention/Transformer service measurements.

Subsequent actual-dispatch diagnostic confirms the untouched eager UnquantizedLinearMethod.apply uses vendor linear instead of explicit mm. Outputs match on diagnostic fixtures, timings differ. A separate new all42-bank version will measure the actual F.linear control; do not present the above as full-service improvement. O128 should be excluded if its apparent gain disappears against actual F.linear. No model/Math/official benchmark/adoption here; fresh full C8/C64 >95% remains required. Published r2 unchanged.
