# BI-V150 32G：K8V16 直接 decode CTA 映射

Local component screen, not an official score. 旧 BF16/warp1 直接映射曾负收益，当前K8V16资源占用改变，单独重新验证，绝不外推旧配置。

仅添加 DIRECT_DECODE constexpr 与 resolver 分支：一个CTA直接对应一个sequence，grid第一维从 n+n//2 减至n。不改writer、K8V16布局、分段、tile、cache hint、attention与reducer浮点运算。逆变换后整个生成模块AST与当前合格reader完全相同，CPU执行实际Producer36分支。

GPU计划：沿用9形状与18次变化输入Graph阶段，各臂全输出逐位对照、独立GPU/CPU参考、全存储/guards/RNG和7组交错事件。此次只测单query非空序列，不声称支持padding/空query、prefill或完整模型。若组件收益成立，必须先补齐真实框架padding、混合和fallback验证，再进入服务与新Math。当前正式best4997.05/Math100of105不变，硬件profile未采集。
