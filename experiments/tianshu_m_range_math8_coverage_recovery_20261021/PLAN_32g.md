# Math8 运行覆盖 CPU 复核

只使用20261021_m_range_prefill_v2_math8_tianshu原始输出，不启动GPU、不重采样。
保留其audit_failed原记录；单独核对105条原始预测与判分，并检查全部旧证据和源文件快照。
C8实际仅M256调用已有prefill INT8，新range fusion实际0层；因此不声称新融合通过Math8。
新控制器用C64 Math105补足新融合路径，必须真实M4096×42层以及C64旧图实际重放，不因本CPU复核放松。
