# 天数BI-V150 32g：严格TopP修复后的低warps扫描

保持已经通过十输入严格CPU门槛和翻转Graph的修复主体不变，比较16K/w16、16K/w8、8K/w8、32K/w8，以及原生adopted/identity。TopK仍回退原8192/4096/w16。原32K/w16/w32有peaked概率误差，不因速度放宽1e-6门槛。此处warp8配置是新的实际编译，必须重新全部严格通过才能考虑服务。

CPU源码同一性/Router类同一性和语法检查通过；尚未GPU运行。只在当前piecewise服务彻底退出且ixsmi空闲后运行，不排队抢占。
