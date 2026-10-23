# Down64 + LM64 W4有条件原始正式4K

仅当本934来源新鲜C8与C64各完整105题>95%、独立全部审计通过（明确C64原audit_failed经单独CPU校正、不重跑GPU），再同场短4K增益>0.5%、spread<1%、TTFT<=control1.01才执行。CPU prepare可跳过尚未出炉的short，但不能跳过完整质量；实际GPU默认强制screen。当前还未CPU预检、没有EXECUTION，不能运行。

原始官方13 serve argv/63环境，原benchmark脚本固定SHA，[[4096,1024,64,256]]，4轮排除首轮，不跑16K。保留所有GU/KV/attention/scheduler/sampling/其他M Down和LM8检查，实际Down42 W4捕获链接主模型图、LM64图外真实native W4 calls>=100、load/native scratch/峰值/存储计入必须通过。旧LM64 W8要求显式被新W4证据替代，不造父记录，不修改模型checkpoint或旧来源。

相对已采用r3历史5950.75 total、TTFT3174.58ms，要求增益>0.5%、后3轮spread<1%、TTFT<=3174.58*1.01；明确历史比较非同场完整AB，并附新鲜paired short证据。两臂short与正式全部计时期间停止后台CPU准备/提交/打包，只轻量观察。所有CPU准备先完成冻结再开始任何计时。失败保留不原样重跑追分，所有原始只在容器、代码/关键结果/报告入GitHub。最迟UTC2026-10-23T15:58（北京时间23:58）停止。
