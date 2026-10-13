# 天数 BI-V150 32g：相同RMSNorm函数体启动参数微测

模型实际hidden2048/BF16，对普通及fused residual RMSNorm，M1/4/8/16/32/64/128/576/2048，共18场景×5配置。只改变线程配置，原Triton函数体、grid和输出语义不变。90项均通过CPU FP32参考（rtol=.016/atol=.016，BF16）及Graph回放输出与同配置eager一致检查；默认和w4s1逐位一致。小M<=64的w8s1在本次12个样本中逐位一致且更快，但不能声称所有输入全局bitwise一致。

|类型|M|默认μs|w8s1 μs|减少耗时%|w8逐位同原版|
|---|---:|---:|---:|---:|---|
|normal|1|3.025711|2.660471|12.071220|True|
|fused|1|7.241144|6.075750|16.094061|True|
|normal|4|3.076500|2.708414|11.964453|True|
|fused|4|7.408990|6.247779|15.673001|True|
|normal|8|3.096163|2.732202|11.755241|True|
|fused|8|8.108029|6.981971|13.888178|True|
|normal|16|3.134577|2.772692|11.544924|True|
|fused|16|9.511875|8.375366|11.948319|True|
|normal|32|3.210798|2.790914|13.077266|True|
|fused|32|12.325875|11.187644|9.234481|True|
|normal|64|3.227836|3.038625|5.861873|True|
|fused|64|15.485519|14.570462|5.909118|True|
|normal|128|3.588433|3.795163|-5.761029|True|
|fused|128|17.977124|17.372096|3.365549|False|
|normal|576|6.587682|8.042942|-22.090621|True|
|fused|576|33.649077|33.747375|-0.292126|False|
|normal|2048|15.699422|21.785085|-38.763610|False|
|fused|2048|84.157171|88.304415|-4.927974|False|

fused计时包含每次input/residual两次相同reset-copy；不将表中时间写成纯kernel耗时。普通norm无input原位修改。每组Graph含20次算子，每轮回放20次，7轮轮转顺序，记录真实compiled metadata。LibEntry缓存key不含显式launch kwargs，微测用原LibEntry.fn相同JIT函数体，确保不同warps独立编译，避免候选实际仍命中旧配置。use_c_extension=false，与正式API未设置USE_C_EXTENSION及源默认0相符。

原版w4/stage3；小M的w8/stage1可进入短服务对照。大M2048普通norm的w2s1虽15.699→14.120μs，但fused变慢且浮点输出有差异，暂不混入同一候选。w1小M显著变慢，不采用。所有CPU pass仅覆盖随机样本，尚需数值边界与真实服务路径验证。若端到端收益明确，再跑新独立Math>95及正式4K；不复用旧质量，也不自动采用micro最小时间。

原始包2271556bytes/9分片，SHA256 d3aed87a8f02eda5beaf3e53a8d102d1f6e807e0e88ea2884ddbe8aca8f4e094。分片/整包、18×5结果、7轮样本及Graph输出校验和实际源快照已本地核验。原件service-results/20261014_tianshu/rms_launch/rms_launch。

正式最好3506.84/Math103属于上一个TopP配置，本轮没有服务成绩。性能剖析：TODO，未取得Graph内部覆盖。安装源、权重和原benchmark不改，沐曦只读。
