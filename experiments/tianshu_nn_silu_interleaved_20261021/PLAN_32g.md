# Gate/Up逻辑交错，单dot免acc transpose

旧single_dot的Up有错误，不采用；此次新目录仅改device两处：W列gather成为gate0/up0/gate1/up1...，acc直接reshape(BM,BN,2)+split，不使用原reshape(BM,2,BN)后trans；同时列tile用真正GROUP映射tile_n，原pid%tiles_n不适用于多个Mtile的group8。两处device和host模式开关反向恢复到旧kernel精确原文。当前NN bank物理bytes/容量不改，无额外bank，仅逻辑列排列加载；可能加载变慢，必须测。

10tile×group1/8×M8/64/128/4096=80配置。M64对照保持原正式two_dot且binary SHA匹配，三输入/权重符号/scale变化，全INT32与投影位相同、M64实际timed输出位相同原融合；实时stream Inductor参考Graph/eager每相位先全位校验。编译/数值失败留档排除，失败完整输出每M仅首份。7轮交替30Graph，CPU实际选点INT64/激活复核。单层热缓存，胜出需42bank及4K/新Math。源门槛先冻结，90分钟或23:58北京硬停。
