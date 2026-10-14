# 天数32g：准备更小的Decode查询行tile

把当前BM16/BQ2改为BM8/BQ1，沿用当前KV16/warp1、B24回退KV32/warp2与B64S8；不是此前的直接序列映射，也不改KV页布局。实际grid根据新BQ重算，producer和reducer的KV tile保持一致。

`verify_cpu.py`通过382组实际Router AST与生产LaunchRouter执行，覆盖B1..64、稠密和两种0/1查询长度、BM16/8。每条逻辑query覆盖一次且reducer tile一致；2个全零查询组合不运行。这替代先前未保存的384组口头计数，当前可复现实际结果为382。

GPU探针尚未部署或排队。计划17例数值、尾部、真实编译metadata、Graph改变query后的独立GPU FP32/CPU参考及AB/BA事件计时；BM8不保证后端支持，编译失败须保留为负例。无模型绑定、Math或吞吐收益结论。
