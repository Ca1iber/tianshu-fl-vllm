# LM64 W4 C64仅CPU审计校正：不重跑GPU，不覆盖原失败记录

原934模型/95实际GPU控制器完成105题，官方101/105=96.19%。全部请求成功，但继承Down W4审计最后还要求LM64为INT8，实际新LM64已按设计W4，故原result是audit_failed，必须保留，不能冒充原实验通过。

新CPU副本完整保留原Down所有bank/hash/内存/capture/模型重放检查，仅明确将C64的LM精度要求改为W4A16，并额外要求真实新LM64 runtime calls>=10、旧W8 LM64为空；不篡改任何父记录/precision标签/计数，不放宽数值质量或性能阈值。适配模块单独绑定，原冻结文件保持原字节。全部原始结果与来源/快照逐一SHA，重新执行完整runtime与105题官方review/prediction/report/分母核对。新CPU_recovery_complete结果在独立目录，附原result SHA、原失败状态、实际GPU934/95与额外CPU审计来源SHA。当前没有新的GPU或benchmark。

只有CPU完整复核>95%及全部运行证据通过后，才继续尚未运行的原版本C8；原模型934/控制器95保持相同。后续完整独立资格门禁必须显式承认C64原失败经CPU校正，不能拿未改的旧门禁直接宣称合格。旧短4K v1从未GPU执行，后续使用明确支持此校正的新短4K版本。
