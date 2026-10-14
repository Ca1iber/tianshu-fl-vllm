# 天数BI-V150 32g：严格TopP 32K/64K tile扫描

七配置、十种输入，原生adopted/identity及修复16K全部通过。32K/w16和32K/w32在B64 peaked原输入第46行严格CPU检查未通过：保留概率0.949998954684501，目标0.949999988079071，缺口1.03339457e-6，超过预定1e-6门槛。该输入不计时，不放宽门槛，不接入服务。其他normal输入B64修复16K3.49058ms、32K/w162.94117ms、32K/w322.91832ms的速度不能掩盖语义失败。

64K/w32与64K/w64在九种纯TopP输入分别OutOfResources线程2048/1024、4096/2048，保留全部18失败。TopK输入实际回退原生8192/4096/w16，不能称64K编译成功或加速。

本地复核350原始events、3520行CPU语义记录、原/翻转Graph、4冻结源码/native与generated源码和TopK AST及实际回退，全部审计通过。仅adopted/identity/fixed16k整体有效，当前正式最佳4091.29不变。模型profile未采集，没有新Math或正式成绩。

归档4233141bytes/17parts，SHA256 68f1823175eecdcd6576aac608aa03a026e62ab28c66fc7e26166cd16ab9b697。raw：service-results/20261014_tianshu/topp_precision_large/20261014_topp_precision_large_tianshu/micro。
