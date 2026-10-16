# 天数TopP几何阈值搜索

待GPU。基于严格actual-global-max修复，只将standalone TopP阈值搜索由算术中点改为正概率边界的几何中点（sqrt(lo)*sqrt(hi)，lo=0回退算术）。p=.95、最大32轮、区间1e-12、严格CPU FP64质量/最小集合/order 1e-6、RNG、TopK回退均不变。比较已验证32K max控制、新32K geo及16K geo，先测固定10输入；失败完整保留且不计时。
