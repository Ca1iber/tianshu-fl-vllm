# BI-V150 32G：exponential标量类型预热，组件通过

## 上版本遗留问题

上一版按B64→B1预热，虽证明采样/RNG恢复，但严格稳定性检查失败：只观察到一个冷调优entry，且预热后仍新增一次调优。旧失败及其原始断言完整保留。

## 已验证原因

实际`LibEntry.key`即使对do_not_specialize整数，也按i32/i64类型区分。N=130560至8355840均属i32，64种N不能据此视为64个独立entry。实测大seed i64与小seed i32使用不同entry；原生随机状态读取为有符号int64，offset跨2^31也切换entry。这是本组件新调优的来源，不据此解释先前所有稀有GEMM shape停顿。

## 本版本方案与落地

在真实`torch.random.fork_rng`内，用实际`set_philox_state`将seed(7,2^32+7)与offset(0,2^32)的四个组合预热。FP32噪声缓冲[64,130560]占33,423,360字节；真实key、编译配置和kernel hash逐次记录。CPU检查直接执行原生key AST，确认四entry及N别名、边界。

使用独立FlagGems数据库和配置/代码缓存，实际路径经核对；没有清除全局编译器缓存，因此不宣称全编译器冷启动。源码检查点42337c1e4f20eb1c0ef35fadfd1b6622f90599d0，测量源码12份冻结。

## 组件结果

| 项目 | 上版 | 本版 |
|---|---:|---:|
| 冷预热时间 | 2.7268秒 | 1.8478秒 |
| 明确覆盖的标量entry | 不完整 | 4 |
| 采样对照 | 33场景/846 token | 57场景/1430 token |
| 预热后新增exponential调优 | 1 | 0 |
| 实际异常恢复RNG | 通过 | 通过 |

本版在64个B值及全部57个采样场景均无新增exponential调优。57场景包括33个global/mixed/all-seeded场景，以及2种seed×4种offset（含2^31-4、2^31边界）×B(1,8,64)共24场景。1430个token与对照完全一致；默认和请求级生成器状态前后相同，真实第三次状态设置失败时也恢复原状态。

上述时间是不同源码/entry覆盖的组件启动测量，不是服务吞吐收益。四entry预热可能改变冷原生首轮autotune所选配置和随机数映射；本轮只证明已测缓存条件下的原生采样与状态等价，不声称与所有未预热冷启动首轮token相同。未绑定引擎，未运行新Math/正式benchmark，未采纳到5101.45版本。

## Profile与证据

TODO：没有硬件profile，尚无端到端收益结论。后续若集成，需要独立服务/质量门槛，不能借用旧Math成绩。

原始根目录`/workspace/benchmark-results/20261019_exponential_key_warm_component_tianshu`；本地同名备份`micro/local_download_audit.json`已通过12份源码、实际entry/编译参数、57场景/1430 token和RNG异常审计。归档4,819,834字节，19块及全包SHA256已核对：`fa6cb82dd928c233d985ebc54674d5f2e950396fb1fd7df297c1419619330b9f`。

结论：组件预热覆盖问题解决；当前保留为待集成实验，服务最好版本保持原样。
