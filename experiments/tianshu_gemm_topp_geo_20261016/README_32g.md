# 天数GEMM与TopP32K几何搜索组合

待服务GPU。官方原始serve argv，默认2048预算/256序列/75预热点、CV/KV16/S8和原采样器。控制原生TopP，候选M2048三项GEMM配置加strict TopP32K global-max归一化和几何中点搜索。3+3轮N64/C64 4K/1K，首轮排除，输入逐轮换seed。只有全10输入严格CPU 1e-6/Graph已复核才启动。新Math与正式验证脚本只做准备，不冒充已运行。
