# 软件8位KV prefill参数扫描结果

来源提交：`30a4812b71ab566b1fb7800a625a8481d3e1e78c`。15配置、两种真实builder/writer/forward场景，每配置两次改变输入后的Graph重放，包含编码writer，7组×20回放。64条全输出数值检查与64条CPU head参考记录、224事件组和85冻结来源已本地核验。没有运行完整模型或Math，未采用。

| 场景 | BF16 ms | 最快Soft8 ms | 相对耗时 |
|---|---:|---:|---:|
| prefill2048 | 0.713234 | 4.118615 | 5.775x |
| chunk2048 | 1.843845 | 11.500081 | 6.237x |

最佳均为BM128/T32/w4/s1。实际编译资源为256寄存器、130 spills；旧BM256/T128/w16为256寄存器、908 spills，原始耗时22.947571/65.196790ms。所有15配置都有spill，但这些记录只说明该路径的编译与运行表现，不能单独证明全部耗时来源。调参无法达到BF16水平，不接入服务。

下一步单独验证prefill阶段先解码到BF16临时缓存，再调用原最佳BF16 attention；decode保持直接读取软件8位。额外解码必须计时，必须检查重放时数据/slots/length变化与缓存边界。

完整原始包 SHA256：`217902b527d6e6b1aa3225e5dad0bcaa25cbf4fe2258dcdd15b2ee7485ea84cc`，4871032 bytes，19分块全部核验。

离线审核：`python3 experiments/tianshu_soft8_prefill_tiles_20261019/verify_downloaded.py --root <downloaded>/micro`。该审核复算事件中位数、参数和源哈希，检查已记录断言；完整GPU张量未序列化，不声称离线重新验证全部输出。
