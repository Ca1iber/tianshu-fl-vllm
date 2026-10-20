# KV V8：移除恒1 K-scale 与成对 V 读取

当前407来源新prefill版本正在正式4K测试；本目录独立，CPU准备不改其运行代码，不并行占GPU。先完成正式测试和已准备的Down组件，再运行本组件。

旧K8VI8Heads V2已经把V_scale乘到P侧，C64/4K仍0.751586ms，比K8V16的0.539952ms慢39.19%。本实验不重复该尺度前移；固定相同布局/实际writer/launch/页表，四路为K8V16、原生V8、V8仅跳过K_scale读取、后者再加16-bit成对V读取。

当前自有writer对软件8位K写入的K_scale恒为精确1。只在明确的此模式让reader用常数1，仍保留原物理K_scale平面与写入，V8三路均8448-byte页、同原生spec/reshape/真实buffer，隔离reader改动。不能把该假设推广到任意per-token K量化缓存。V_pair要求D128、末stride1、其它stride偶数，用uint16 load后两个signed int8还原再join，不改变缓存数值或原P/V尺度算术。

CPU验证全部65536个word的有符号字节解码和0..127维映射；8个真实原生连续/strided布局及全raw平面边界；writer与已验证V2字节完全相同；生成代码相对V2精确3处可逆修改，API/reducer/cast AST不变，默认false保留原路径，CPU未初始化GPU。

GPU计划复用原10场景×3变数据/slots/长度相位、完整writer+reader+reducer计时，7轮轮换顺序各20 Graph重放；全原始物理存储、输出/scratch保护、CPU/GPU RNG、不变输入、GPU完整参考及CPU抽样、真实writer独立再量化核验。编译metadata明确记录两个constexpr开关和实际regs/spills/binary。未跑GPU，不宣称收益、模型正确率或新正式成绩；仍需新模型profile和全新Math才能采用。

CPU执行必须显式 `CUDA_VISIBLE_DEVICES=-1`。最初可见GPU的vLLM导入会初始化CUDA上下文，新增is_initialized断言发现并拒绝；原失败日志保存在容器 `20261021_k8vi8_fixedk_pairv_CPU.log`。隔离重跑 `20261021_k8vi8_fixedk_pairv_CPU_invisible.log` 全部通过且未初始化GPU，代码不做设备计算。此导入检查发生在正式4K排除的首轮期间；后续CPU检查使用不可见GPU进程。
