# Spec 验证GQA tile的服务级隔离确认

同一Spec、固定7步、target图/draft eager、4K输入/128输出/C64/N128三轮排首轮。仅将target causal小query批次2D M16/KV32/w4/s3改为已检查的M128/KV64/w8/s2，长Prefill和draft非因果路径保持原tile。

基础Spec均值3187.994184→3200.421505 tok/s（+0.389816%），TTFT约59064.77ms。候选三轮1398.666561/3199.247976/3201.595034 tok/s，冷轮386.562470s、1条调优，后两轮零新增调优，共384计时请求成功。算子−62.07%没有转化为同幅度服务提升，单独收益小；不采用该单项版本，也不为它启动数学/正式benchmark。

完整备份13074178 bytes、SHA42252ff22fa316f568b5f6cf965dc3b8dac3cf38c60dfa40fc7cbd3abc5e942e，已逐块及整包本地验证。继续隔离接入既有Prefill优化，再使用1K输出检查decode部分。正式最好成绩保持2712.55。
