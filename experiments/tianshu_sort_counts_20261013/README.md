# 天数32g 局部直方图计数位宽

只读参考用户沐曦ops 3caa4fc0，在天数自有实验中测试局部acc64/32/16/8。保持浮点排序值、global计数、int64索引和稳定排序。真实循环每lane最多8，窄归约升int32，signed128回退int16、32768回退int32。

80项histogram独立CPU参考与120项完整稳定排序/图重放全部通过，真实M1/M32/M64词表与尾部形状覆盖。M64随机升序完整sort：64 24.946429ms，32 23.508922ms（速度+6.114732%），16/8没有额外收益。TTIR窄acc真实存在，但物理寄存器与32均143。

真实服务双组五轮N64/C64/4K1K最后三轮：原版3253.050661 vs int32 3245.940231tok/s（-0.218577%），640请求成功、7greedy相同。局部收益没有转成服务收益，不采用，不重复正式benchmark或Math。报告和完整原始SHA位于results/20261013/micro_32g及service_32g。

hooks/sitecustomize.py、run_service_pair.py与native/run_service_trial.py的计数off/32默认计算路径已在上述完整服务对照中验证；16/8服务和新的watermark/input-length参数尚未测试，不能当作已通过。所有安装源和原benchmark未改，沐曦只读。
