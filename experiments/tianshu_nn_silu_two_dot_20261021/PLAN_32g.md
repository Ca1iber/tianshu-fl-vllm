# 双dot完整NN组件：补齐主批次与stride，不借用失败单dot

原32组在4个M8 two_dot后因single_dot整数错误停止。单案例完整CPU98304 INT64确认Gate正确、Up43008错误，非浮点微差；连续性提示未修复，cat默认有编译限制。原源/失败/数据保留，单dot另寻替代加载方式，不因错误候选停止其他合法候选探索。

本轮使用字节相同原kernel.py，只选择two_dot；M8/64各连续和行stride2055，共4输入形状×bn64/128×warps2/4=16配置，bk128、stages1，actualM不补齐。包含4个旧M8点共同对照以及此前未跑的12个shape/config；新时序不挑选旧最好，只完成同一次完整16配置组件。目标是补齐主批次证据，不是将原32组失败改成成功。

其余验证、真实layer0权重/NN物理布局/原选优NN GEMM基准、3变输入Graph、全计时BF16逐位输出、完整诊断INT32与投影、独立checkpoint行/INT64/量化/激活CPU参考、边界/输入/RNG、7组轮转×20replay全部沿原。GPU后离线重读完整输出。16配置探索最低值不是模型成绩或独立holdout。FlagGems激活组件不自动等于实际Torch编译模型activation。

只有主批次收益才考虑FX静态8/64局部融合，prefill保持原路径；原NN policy capture里要求旧NN launch，需要显式适配实际新融合capture证据，不能写虚假旧launch记录过门。尚无模型hook、新Math/benchmark，不改正式5586.28/102of105。

1800秒与23:58硬停，冻结源，保存所有结果。完整raw仅容器，GitHub代码/报告/关键gate，本地无副本。
