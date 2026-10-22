# r3后GPU空闲窗口：仅清理可重建FX pickle cache

GPU空闲、全部r3源和控制路径确认无/tmp/torchinductor_root/fxgraph输入，且该目录无Python源码后，946个FX文件逐项SHA登记；精确tar备份到容器/dev/shm/tianshu-storage-after-r3-20261022/fxgraph_backup.tar（1151262720bytes），tar逐项比较、源文件SHA再次验证后才清理其中内容。原生成.py、模型checkpoint、raw结果、strictTritoncache和源码全部保留。

可用空间4471832576→5621985280bytes，恢复1150152704bytes。清理后r3启动dry-run再验证903模型源/73控制器，并重新检查384个strict缓存文件全部SHA，均通过，没有启动GPU服务。tar和完整946文件清单在容器，不上传缓存/日志/数组；GitHub保留关键结果和备份SHA。该FX缓存可以由编译器重建，归档位于容器共享内存，不视为长期评测数据备份。随后才启动M64组件，和服务不重叠。
