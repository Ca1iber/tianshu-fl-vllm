# 首次模型接入失败：绝对导入命中同名pass

冻结633f4f9，2026-10-23 13:26启动C64。13:27 Worker初始化失败：旧NN融合policy在`from fx_pass import FusionPass`误导入新服务目录的同名模块，3参数旧构造调用落到2参数新构造。尚未加载checkpoint、开始Math请求或执行新融合数值kernel；C8未启动，不能报告正确率或吞吐。

原始失败保留于`/workspace/benchmark-results/20261023_prefill_RMS_QKV_math64_tianshu`，原模型/控制器冻结文件不修改。独立v2将新pass改名prefill_qkv_fx.py并取消插入新BASE到sys.path，再完整CPU准备和冻结。此修正不改数值内核、不放松任何质量守卫。GPU已恢复空闲；正式最好仍r3 5950.75 total tok/s。
