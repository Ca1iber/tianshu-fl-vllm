# 连续采样行：gather 与 view 的数值检查

真实FlagGems index算子在B1/2/3/4/5/6/7/8/16/64、7B query rows、H2048 BF16的连续/非连续输入20个场景中，与CPU参考和prefix view全部逐位相等，源数据不变，view共享原始指针。因此复制语义相同。

实际两个96-token smoke请求均成功，但greedy后续文本与之前非view Spec不同；这里不证明全序列等价，也尚未证明具体数值分歧原因。只证明行复制正确，尚未采用；是否有服务收益和质量仍需后续独立确认。完整备份1736405 bytes/SHAfb56b2502447158e03b32cb0db4a2424ca99120be4c3fe883377c0c7ade44f52已本地完整核验。
