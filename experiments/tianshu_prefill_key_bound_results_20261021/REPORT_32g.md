# Prefill maxK收紧：24个GPU场景全有效位相同，仅证明组件

已完成独立GPU组件：八个纯prefill/混合/前缀/长上下文边界场景，乘三种真实K8/V-BF16共享底层storage步长12288/12544/24576bytes及不同offset，共24。原生未改BoundedDensePrefill和gather_decode，隔离进程仅将arena设为与当前profile相同的8192pages。每臂每个有效K/V位都与CPU独立解码/原V相同；原/收紧两臂有效位SHA相同，全部无效token、末尾保护page保持17，毒化的无效物理page索引被mask，dense ID表精确。源SHA仍匹配CPU门禁。完整结果含全部逐臂元素计数、SHA、寄存器信息和事件组，不含模型回答或大张量。

|场景，stride12288|原页数→收紧页数|原gather ms|收紧gather ms|
|---|---:|---:|---:|
|C8等长纯prefill|256→256|0.012061|0.012041|
|63decode+短prefill|2048→5|0.057509|0.003972|
|32decode+32prefill|4096→288|0.115371|0.013676|
|62decode+带前缀prefill|4096→514|0.116484|0.020066|
|60decode+4prefill|8192→2048|0.253988|0.089970|
|C8纯prefill前缀|1024→1024|0.048061|0.047777|
|57decode+7非对齐prefill|7168→532|0.198748|0.019087|
|整批131072边界、prefill2404|8192→151|0.226827|0.008485|

七组交替顺序CUDA事件，每组3次回放、每个独立组件Graph16次实际gather，取中位数，排除Python发射空洞；不是模型Graph路径证据。对照与候选同一真实输入数据，差别只有暂存矩形和grid。其他两种步长同样完成；详见KEY_TIMINGS。无需收紧时，微小时差应视为噪声；不要平均人为选取场景的加速比，也不能将14倍或26倍gather比率外推为模型收益。

CPU元数据辅助函数2048分布/119960逻辑页地址先通过；单独policy的12项实际原生Route AST接入检查也通过，preserve原Route/cu、decode参数及其他dense实例。模型尚未安装此hook、没有该变更的Math或吞吐实测。未来仅prefill stage副本收紧maxK，绝不改dispatch的decode全局maxK；完整key长度包含前缀。先arena8192做独立模型AB，再考虑单独arena4096容量因子。GPU验证期间无其他GPU服务；之后才启动staticcompile的独立4K对照，二者未组合。

原件：/workspace/benchmark-results/20261021_prefill_key_bound_component_tianshu；源码：../tianshu_prefill_key_bound_component_20261021。当前推荐5586.28/TTFT3780.63版本不变。
