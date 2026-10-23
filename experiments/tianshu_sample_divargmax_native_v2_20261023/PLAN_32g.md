# 原生 argmax 循环内融合采样除法

父项独立两段融合在普通输入有组件降幅，但首个NaN case与原生token不一致，原失败不改。这里是不同候选：在实际 FlagGems argmax_kernel_inner 原源码的三个load位置加入同样 FP32 `/` 和完整score写回，原循环/归约/启发式/NaN行为完整保留，不采用自定义首NaN规则。新增Q参数，原argmax函数其余AST不变；生成来源SHA固定，原文件不改。固定使用原启发式，不重新挑配置。

组件覆盖与父项相同的8case×2phase；完整scores逐位、native token逐位、noise/base/RNG不变，保存再断言，保留任何失败。CPU独立检查：有限完整商容差2e-6，token按原后端实际tile循环规则精确匹配；额外同时记录标准NumPy argmax的NaN差异，不隐瞒原后端已有NaN行为。strided测试包含显式contiguous及写回成本，实际服务仅考虑contiguousFP32输入。主计时7轮交替、每轮30图重放，输入恢复两边等量，剔除首轮。若无正组件不接服务；若正组件还要真实RNG和服务/Math验证，不把组件计时当最终模型成绩。硬截止23:58北京时间，最多600秒。

启动修复：上一原循环候选首次融合调用因参数位置绑定错误，在启发式解析阶段KeyError(N)，未执行融合kernel、未完成case或计时。原件保留。本v2仅把调用改成明确具名inp/noise/out_index/M/N；生成设备源码SHA完全相同，阈值、用例和计时不改。启动前增加AST参数绑定检查。
