# 自定义两段除法argmax：NaN语义失败，不采用

完整普通/strided/B8/B1/并列值10阶段检查已通过并保留；连续C64组件0.53960→0.46306ms(-14.18%)，strided含相同输入恢复0.99091→0.48052ms(-51.51%)。随后首NaN用例失败：完整商bits仍相同，但自定义首NaN argmax与实际原生分块循环不同。整体phase=failed，没有模型接入/Math/benchmark；不能把前面的有利计时算成通过方案。该失败不通过删NaN case或放宽原断言“恢复”。后续原生循环融合是不同设备实现，见相邻native_v2实验。

原始结果和已保存输入/完整输出在 `/dev/shm/tianshu_sample_divargmax_20261023`，完整日志在 `/workspace/benchmark-results/20261023_sample_divargmax_runner.log`。NaN失败发生在保存该阶段输入之前，因此不声称原失败NaN数组完整落盘；源码、异常和已完成10阶段原件保留。持久归档路径另附。
