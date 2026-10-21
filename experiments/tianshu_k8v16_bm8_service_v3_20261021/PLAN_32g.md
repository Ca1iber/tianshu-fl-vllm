# BM8+直接映射接入现行471源：M64单项对照

组件14场景×两相位，112完整输出、8044544元素、304独立CPU GQA向量全部通过且相对current全部逐位相同。C64/4K .552524→.528048ms(+4.635%)、5K .682641→.653482ms(+4.462%)，实际M8Tile167regs/0spill；BM8单独负收益，直接映射BM16仅+.5%，必须组合验证整模型。
原471数值源逐字节保持。只在实际3D num_seqs64使用BM8/BQ1和grid64×2×S4及新解析器；其余批次与2D完整调用原pair_rows函数，不启用QKV NN，不增加packed权重，不改变codec/归约/采样/预算/seq_limit。变化发生在实际Graph捕获前，要求42层捕获及实际C64请求replay，实际source/config/compiled metadata必须绑定组件。
旧pair记录描述基础选择器和进入子选择器前的参数，不能误当最终source/grid；新记录提供实际JIT源/参数/二进制哈希，审核明确覆盖旧active字段的作用范围。
reference直接现行471入口；candidate新源图，原13参数/62配置归一化仅PYTHONPATH不同，相同字节客户端/输入seed。4K/1K C64/N64三轮排首轮，稳定>.5%才新Math C8及实际C64，然后正式4K N256四轮；不常规16K。失败不覆写，不重采样追分，23:58北京硬停。
全部项目资料留容器及GitHub，完整响应/trace/原始张量只留容器。通过性能前不改变正式5695.95推荐版。

V1在worker构造时的inspect.getsourcefile(class)校验失败：旧pair策略模块没有注册到sys.modules，inspect误认为内置类，模型与新kernel尚未执行。V2只改为读取该类__getitem__方法的co_filename来源；CPU实际导入同一未注册模块复现类检查失败并验证方法检查通过。V1所有源码与失败原件保持，新模型数值代码/组件选择不变，复用已完成直接471对照。

V3绑定真实模型ABI重测的组件。V2与旧组件SME不同而拒绝，现411/18a1同键下完整ELF字节仍因.debug_info不同而SHA不同；BINARY_CODE_GATE逐节证明仅非加载.debug_info有变化，全部其他节(包括.text/.rodata/.note/符号表/其他debug)及ELF执行头完全相同。新selector保留完整SHA供审阅，仅忽略这一项非加载debug_info；模型所有可加载代码/数据与符号布局仍须精确等同实际计时的组件，不接受未知binary代码。该校验helper/CPU Gate加入冻结图；原模型和组件失败全部保留。
