# BF16大M：group1/8 × 大tile × W重排

此前六tile均固定GROUP_M8；M4096三个role均未胜当前库，M2048 O/Down special保留后多数旧收益消失。新轮加入BM256 BN256 BK32 W16等现有M2048实际tile，并将group1 row-major与group8各自独立；7tile×2group×12大M×3role×2相位共1008相位。仍按相同tile重排对照分离W地址与program调度，原生/当前实际M2048special保留。所有候选维持.02+.01abs全输出GPU在线、CPU真实fixture选点F64，7轮交替×20Graph。GPU空闲串行执行，最多90分钟或23:58北京时间硬停。单层热缓存筛选，采用需42bank及真实4K，不能仅以不同tile比较宣称W重排收益。
