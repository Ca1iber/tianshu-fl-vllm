# 天数逐层静态INT8 KV与scale外提：仍无组件收益

3场景均使用当前.cv/page16/B64S8；手工固定逐层power-of-two scale=1/16。INT8_native_tensor借用原生per-tensor反量化分支的数学语义，但存储明确是INT8，并非FP8。folded将K scale并入softmax_scale，V scale只在分段归约后的FP32结果乘一次；producer源AST完全相同，reducer两处新增可逆且默认分支保留。BF16控制沿原路径。

|B|最大ctx|变体|GPU ms|耗时变化|producer寄存器|
|---:|---:|---|---:|---:|---:|
|8|33|BF16|0.015970|+0.0000%|88|
|8|33|INT8_native_tensor|0.016699|+4.5688%|158|
|8|33|INT8_folded_w1|0.016294|+2.0339%|158|
|8|33|INT8_folded_w2|0.017412|+9.0304%|155|
|64|4096|BF16|0.850825|+0.0000%|85|
|64|4096|INT8_native_tensor|0.959981|+12.8294%|158|
|64|4096|INT8_folded_w1|0.905013|+6.3689%|158|
|64|4096|INT8_folded_w2|0.907665|+6.6806%|155|
|64|5120|BF16|1.056704|+0.0000%|85|
|64|5120|INT8_native_tensor|1.195164|+13.1031%|158|
|64|5120|INT8_folded_w1|1.126175|+6.5744%|158|
|64|5120|INT8_folded_w2|1.124080|+6.3761%|155|

B64/4K最好的INT8外提仍慢6.37%，5K慢6.38%至6.57%；未计真实量化writer已无收益，不做框架整合/新服务/Math或正式benchmark，不采用。外提减少了相对逐tile scale运算的成本，但未快过BF16。资源记录BF16 85 regs、INT8 155/158 regs，spill均0；这不等同于已证明寄存器占用是唯一瓶颈，没有性能计数器因果证据。

两阶段改变Q/K/V与量化数据、先清空output再Graph重放，6阶段均通过自身dequantized FP32参考与CPU抽点，output/segment scratch尾部不变。所有本次随机fixture的K/V clipping为0。相对原BF16参考的近似误差另存，并非逐位相同或模型质量结论；当前BF16最佳Math102/105不用于量化候选。完整GPU Tensor未序列化，本地核验检查记录、64快照、84事件并重算中位数/变化率，重新生成reducer字节与producer AST亦通过。

raw全包4268437 bytes、17分片SHA256 9171262086d2873f460e69469a7f72f734b044a888dba5d63147ef35c69796bb，分片/整体均verified。见service-results/20261016_tianshu/kv_int8_tensor_attention_micro/20261016_kv_int8_tensor_tianshu。冻结probe/helper保持原字节，无安装源码/权重/原benchmark/最佳启动器或沐曦改动，GPU已确认空闲。
