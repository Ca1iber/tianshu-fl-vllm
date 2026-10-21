# paired-load 诊断未完整结束

前三路日志分别为双dot正确、原single整数错误、增加max_contiguous提示的single仍有相同错误；最后分别连续load后tl.cat的变体在编译期失败，原始实验phase保持failed。

`FAILURE_LOG_32g.txt` 保存原始输出，末尾为 tl.cat lowering AssertionError。控制器直到全部变体成功后才保存张量，故本轮没有stages.pt/result.json，不宣称独立CPU验证，也不凭前三条GPU比较补写成功gate。原始目录 `/workspace/benchmark-results/20261021_nn_silu_load_debug_GPU_tianshu`。

后续采用不同的新目录join实现以保存每次失败完整实际输出；该实现正确但慢，见 `../tianshu_nn_silu_join_20261021`。双dot完整合格结果见 `../tianshu_nn_silu_two_dot_20261021`。
