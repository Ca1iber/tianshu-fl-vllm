# 分组GEMM：融合乘加舍入差异已确认

新诊断重现初版相同layer0权重、M64输入种子和原始编译二进制；没有改写初版失败记录。完整整数结果由native、instrumented Triton和CPU精确整数模型相互验证一致。分步FP32乘加与FMA的差异如下：

| 分组K | BF16输出数 | 对分步舍入不同 | 对FMA不同 | w2/w4 |
| --- | ---: | ---: | ---: | --- |
| 128 | 131072 | 17 | 0 | 相同 |
| 512 | 131072 | 6 | 0 | 相同 |

设备汇编实际含 `ml_fma_f32`；在本环境中诊断版本即便写成 `mul_rn`+`add_rn`，生成的指令和FP32结果仍表现为FMA。原始G128_w2的binary SHA256为 `f22f6631532b1ddae814246131f947bb128fe6dcbc3068796149ea2332cf66e1`，与初版完全一致。这里报告本环境实测，不泛化为其他设备或编译器版本。

GPU诊断保存每组INT32、未乘WS的FP32累加器、最终FP32和BF16输出。CPU独立后处理用INT64重算3768个分组点积、对119个输出按每一组调用系统`libm.fmaf`，包含全部BF16差异位置及随机点；均匹配FMA模型和原始输出。所有实际Source/snapshot与模型相关权重哈希通过检查。后处理程序没有导入Torch/Triton/vLLM或使用GPU。

这是诊断结论，不是候选数值资格或性能结论。原始诊断GPU程序没有完整记录输入/RNG快照检查，因此本报告不声称其通过这些运行守卫。独立后处理位于单独目录，GPU冻结源码和原始数组未修改；详细数组、LLVM/device assembly仅容器 `/workspace/benchmark-results/20261022_down_group_diag_v2_GPU_tianshu`。

后续新版本明确FP32 FMA累加规范，用native逐组整数点积加独立FMA归约作为完整GPU参考，再用CPU INT64+libm.fmaf逐点验证，保留严格比特一致性而不放宽容差。初版失败仍为失败；发布r2不变。
