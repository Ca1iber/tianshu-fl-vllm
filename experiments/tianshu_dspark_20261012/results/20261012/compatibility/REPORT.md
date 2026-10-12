# DSpark 兼容诊断检查点（13:45）

已完整保留并备份6轮兼容smoke（包括5轮失败）、原始target对照、query-only RoPE失败及修正后的数值检查、原始固定前缀对照。安装vLLM0.24没有DSpark注册，使用独立进程兼容类和DFlash传输，7个query包含anchor；逐步采样使用Markov bias，原始target rejection sampler保持不变。原模型权重及配置只读，尚未采用。

原始短序列2D注意力存在独立数值错误，显式无sink分母L0修正5/5通过；服务双方统一使用修正后，第一个提示的8个greedy token完全一致，第二个提示前6个一致，第7个token不同。第二提示的基线` of`与`,`的logprob完全并列（−1.615134716），Spec版本两者相差0.125。两提示首个token的前10候选logprobs逐项相等。后续target logprobs仍有数值差异；此证据不证明整个序列完全等价。

原始S16 decode在K13/14/15/16/21/23/30的7/7场景通过FP32及独立CPU检查、输出/scratch尾部保持，因此保留原始3D decode。draft RMSNorm及query-only RoPE扩大到实际16个query heads后18/18 CPU及尾部检查通过。

使用原始补全文本（未套聊天模板）的96-token请求仍存在greedy文字差异。该诊断不作为数学正确率；最终独立Math-500 Level3需105项完整、零请求/评分错误，正确率严格>95%，不以逐字匹配作为门槛。

现在运行同参数的eager C8/N16/4K128三轮短对照，双方使用L0/default native tile，排首轮并记录计时轮调优。该对照不等于已验证最佳的图模式优化组合，也不作为官方benchmark。随后可对投机验证阶段的GQA tile和target图模式继续实验。

历史完整备份3073221 bytes、SHA4e9c7d2e1e59c67742d49b3089e31a0b95d4373d4ffc3d3498685eebcced27c0；L0后续完整备份2565817 bytes、SHAcc329a37cb8413740c31b4d411d115f10afaf8efed67c3366107cf49a02bf485，均已本地逐块及整包核验。当前正式最佳2712.55 total tok/s / TTFT4720.38ms / Math102/105保持不变。
