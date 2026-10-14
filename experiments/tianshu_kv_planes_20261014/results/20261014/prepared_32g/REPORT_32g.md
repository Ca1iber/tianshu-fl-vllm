# 天数32g：K/V平面布局组件实验准备

在页长16及B64S8基础上，对照每页交错的K/V与整层分离的K/V平面。逻辑cache shape保持[num_blocks,2,16,2,128]，只改变物理stride；KV writer与attention仍用实际后端实现。框架提供get_kv_cache_stride_order接口，后续可进程内返回(1,0,2,3,4)，此检查点尚未修改或启用该接口。

复用已通过的真实builder/KV writer/forward组件验证方式，新建13例：Prefill、chunk、ragged、mixed、非Decode排序，以及B8/16/24/32/64 Decode。两组相同逻辑KV、独立随机页表；两轮Graph中就地修改Q/K/V、slot_mapping与长度，比较FP32/CPU参考、写入整个物理存储及padding/segment尾部。7次AB/BA设备事件时间仅用于算子筛选，不作为服务吞吐或质量结论。

源码已通过AST检查，GPU结果待定。串行控制器设计为等待当前page16/32服务三轮对照成功退出、原始归档完成、GPU闲置和端口释放后，才运行一次组件探针。它不自动采用布局，也不自动启动完整benchmark。截止时间继续服从现有预约文件。没有16K服务测试、安装源码改写或沐曦修改。
