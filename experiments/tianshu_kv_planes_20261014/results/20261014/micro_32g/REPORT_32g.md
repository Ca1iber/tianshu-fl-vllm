# 天数32g：分离K/V平面没有明确收益

页长16/B64S8下，真实builder、KV writer、hybrid/native attention对照每页交错K/V与整层分离平面。13场景全部通过GPU FP32、CPU抽点、全物理KV存储、output/scratch尾部与两阶段改变输入的Graph重放检查，两组输出逐位相同。

所有组件耗时变化在−0.617511%至+0.407437%之间；Prefill2048 +0.077964%、chunk2048 +0.010810%、mixed64 +0.378390%、Decode64 −0.065403%。没有明确收益，不准备整模型stride-order替换或服务benchmark。框架KV stride接口保持原样。

本地独立重算每例7次AB/BA设备事件中位数及变化百分比，核对两组cache block stride比例2:1、14份冻结源快照及全部数值/尾部记录。存档是确定性输入代码与标量检查指标，未序列化完整GPU Tensor；不把这些指标扩展为模型正确率。

原包2,940,539bytes/12片，SHA `c34e29a5122bbd0c48072922912ee9176dd179884ed50c27f136d4743e1d1fac`，全包/分片SHA核验。原件service-results/20261014_tianshu/kv_planes_mixed/kv_planes_mixed。计时见timings_32g.csv。

当前正式最好仍3826.24 total tok/s，标签tianshu-best-kv16-b64s8-20261014。没有完整模型测试、新Math、16K或安装源修改。
