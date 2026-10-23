# LM64 W4新鲜C64：101/105，CPU完整审计校正通过，原失败保留

同934模型/95实际GPU控制器完成全105题，官方101/105=96.190476%，全部105正常stop、0请求错误，无cap或遗漏；错题87/42/18/23。尚未运行过本版C8或吞吐。原controller在后处理运行审计中失败，result仍为audit_failed；失败点是继承Down的最后一条还要求LM64 INT8，但本模型已按设计改W4，此条遗漏适配。

新的独立CPU副本仅将该LM精度要求替换成实际W4A16，同时新增真实LM64 W4 calls>=10、旧LM64 W8调用为空的要求。Down原42bank/全部SHA/真实profiler和scratch/峰值/42真实capture及C64主模型图链接逐项完整保留；GU/KV/scheduler/sampling/attention/LM8/原所有其他精度审计全部重算通过。实际LM记录至少512次W4调用，图外执行，不冒充Graph重放；新42+1真实内存探针计入。完整原934来源/95控制器及其快照、105reviews/predictions/report/raw SHA和原checkpoint5GB SHA均重新核对。

原文件和冻结来源均未改，原实验仍标为未通过；新result/accuracy在独立/workspace/benchmark-results/20261023_dots_LM64_C64_CPU_recovery_tianshu，phase=CPU_recovery_complete，显式绑定原失败result SHA和单独CPU校正代码SHA。不重跑C64、不造父记录或改precision标签。C64_RECOVERY_GATE为部分资格，model_quality_qualified=false，C8待新鲜运行。现通过完整CPU校正后继续原同934/95版本尚缺的C8；旧短4K v1未GPU执行，完整质量与后续短测将显式处理CPU校正，不把原failed当原complete。
