# 天数BI-V150 32g：严格TopP低warps没有改善16K/w16

六配置、十输入及翻转Graph均通过严格CPU1e-6门槛，包含BF16 normal/peaked、ties、constant、masked及TopK原生回退。420events、4224行语义、四冻结源/native/generated/TopK AST本地全部复核，无编译错误。

B64 normal：当前修复16K/w16 3.513763ms，16K/w8 4.426179ms、8K/w8 5.683052ms、32K/w8 5.463763ms。B64 BF16 normal分别3.067372/3.742477/4.499908/4.513498ms；B64 peaked和BF16 peaked同样更慢。32K/w8严格数值通过，不沿用32K/w16/w32的失败判断，但速度无收益。

不进入服务，不采用。完整原始4473307bytes/18parts，SHA256 e0064756e1f7fa53cfc173baac3d014402dbcb8762749f916c55af3b0bb7f015。raw在service-results/20261014_tianshu/topp_precision_lowwarps/20261014_topp_precision_lowwarps_tianshu/micro。未采集模型profile，无新Math或正式分数，当前best4091.29不变。
