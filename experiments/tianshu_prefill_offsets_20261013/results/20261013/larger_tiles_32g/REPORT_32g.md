# 天数32g：未覆盖的Prefill tile组合

M512/KV128和M256/KV256均需要196608 bytes共享内存，M512/KV64需要163840 bytes；Triton编译器报告此设备本次kernel限制131072 bytes，因此这些配置未运行，不是数值失败。M256/KV128/w8/s1四场景FP32/尾部均通过但明显比w16/s1慢。保持M256/KV128/w16，stage1→2四场景都正确，耗时约降低9.7–10.5%，进入独立扩展数值验证。只用旧地址64与No-sink L0 body，未改变已验证服务。

|Q|KV|stage1 ms|stage2 ms|
|---|---|---:|---:|
|[2048]|[2048]|0.687182|0.620652|
|[2048]|[4096]|1.813910|1.624125|
|[2048]|[16384]|8.494559|7.619746|
|[1537, 511]|[16384, 4096]|7.855681|7.057539|

完整raw2154536bytes、9分片及整包SHA256 a413caa681991f3efefd609836f174fc870ec1eba9e01f79fa45060bd47a1a86全部本地核验。执行source/helper/reference/generated源码、错误、FP32/CPU/逐位检查和全部交替计时保留在service-results/20261013_tianshu/prefill_larger_tiles_micro。probe逐字匹配真实快照，未改安装源码或原始benchmark。
