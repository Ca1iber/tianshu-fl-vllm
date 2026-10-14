# 天数BI-V150 32g：K/V不对称缓存策略与逐出提示

当前已采用的CV/83/S8/KV16/w1内核为基线，独立比较K.cv/V默认、K默认/V.cv、K.cv/V.cg、K.cg/V.cv及cv两载入不同evict_first/evict_last组合；cv/cv无逐出提示为identity。保持算术、分段、grid、tile、地址宽度完全不变。生成源码可逐字反向还原8deb当前CV主体。

沿用17场景的CPU/FP32、原/变QKV Graph、输出及scratch尾部检查和九轮轮换events，新增策略可能编译失败须保留。比特一致是采用的必要条件；本地算子试验不是服务/Math/正式成绩。未部署运行。
