# 性能结论暂撤回：Graph stream检查

数值3024 eager场景及CPU147456实际GPU选点仍是有效已执行检查。后续新tile扫描的动态Graph phase1发现Inductor激活参考滞留旧输入，原因疑似两轮都在开始处缓存stream句柄。原42-bank计时Graph可能缺少激活，并且中间Down可能读取上层旧激活；所以原报告所有timing/gain（含M384+3.92%、M4096-7.51%）暂撤回，不能采用为性能结论。

原来源、result和CSV均保留；后续需实时getCurrentRawStream、全42层Graph跨输入对比eager、整个链Graph激活poison检查，然后新目录重测。正式4K 5618.01与先前FlagGems-only/BF16扫描不使用该显式固定stream，不受这项问题影响。
