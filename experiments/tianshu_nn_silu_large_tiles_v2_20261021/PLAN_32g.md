# 宽tile v2：实时stream和三输入Graph/eager一致

原宽tile轮由fixed default stream导致Inductor参考Graph未捕获激活，全部phase1错误已保留、PID中止，不能归因候选kernel。独立stream三相位诊断已确认因果。v2仅修改生成激活每次实时取current raw stream，并在每配置/每动态输入相位完整检查reference Graph激活与独立eager位相同后才对候选评分。240配置、三动态相位、M64当前原版binary强制匹配、全INT32/投影/量化/CPU整数与阈值保持。失败全输出每M仅保留首个，避免无意义重复占空间。其他全部同原轮，源GPU前冻结、90分钟或23:58北京硬停。单层热缓存筛选，采用需42bank/真实4K/新Math。
