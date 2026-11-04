# CDNA3 (gfx942) BF16 GEMM：用 FlyDSL 从 v0 写到 v2

参考：

- [FlyDSL kernel tuning guide](https://github.com/ROCm/FlyDSL/blob/main/docs/kernel_tuning_guide.md)
- [gfx950-gluon-tutorials / gfx942 / intra_wave](https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/gfx942/intra_wave)：v2 的设计来源

问题：`C[M,N] (fp32) = A[M,K] (bf16) · B[N,K]^T (bf16)`，A、B 都是 K 连续，`M=8192, N=9728, K=8192`。
`M/256 × N/256 = 32 × 38 = 1216 = 4 × 304` 个 CTA，正好是 304 个 CU 的整数倍，所以不讨论尾块。

| 文件 | 内容 |
|---|---|
| [v0.py](v0.py) | 1 wave/CTA，直接从显存读到寄存器做 MFMA |
| [v1.py](v1.py) | 4 wave/CTA，A/B 经 LDS 共享，XOR swizzle |
| [v2.py](v2.py) | v1 + 单 wave 软件流水（4 region、寄存器预取、`sched_group_barrier`） |
| [bench.py](bench.py) | 正确性 + 计时（含 torch.matmul / hipBLASLt） |
| [slides/](slides/) | 培训 PPT：`make_figs.py` 画图，`make_ppt.py` 生成 `CDNA3_GEMM_FlyDSL.pptx`，`measure.sh` 采数据 |

```bash
python3 bench.py                    # 所有版本：正确性 + TFLOPS
python3 bench.py v1 v2              # 指定版本
```

---

## 读前须知（给有 NVIDIA 经验的读者）

本文假设你熟悉 CUDA 上的 GEMM 优化（tiling、smem、swizzle、软件流水）。AMD 的硬件概念不集中讲，而是在 v0 → v1 → v2 里第一次用到时引入：

- 每个版本开头的 `x.0` 小节只讲这一版需要的硬件知识。
- **NVIDIA 对照** 框把它和你熟悉的概念对上：哪些一样、哪些不一样、为什么。
- 完整的规格对照见[附录 A](#附录-acdna3-与-hopper-完整对照)，常见误区速查见[附录 B](#附录-bnvidia-经验在-cdna3-上的常见误区)。

**术语映射**

| NVIDIA | AMD（CDNA3） | 首次出现 |
|---|---|---|
| SM | CU（Compute Unit），每 CU 4 个 SIMD | §1.0 |
| warp32 | wave64 | §1.0 |
| thread block | workgroup（本文仍叫 CTA） | §1.0 |
| 寄存器 | VGPR（向量）、AGPR（累加器）、SGPR（标量） | §1.0 |
| CuTe layout / CuTe DSL | FlyDSL layout（`make_ordered_layout` 等） | §1.1 |
| `mma.sync` / `wgmma` | MFMA（`v_mfma_*`） | §1.2 |
| SASS（`cuobjdump -sass`） | gfx942 ISA（`FLYDSL_DEBUG_DUMP_ASM=1`） | §1.4 |
| Nsight Systems / Compute | `rocprofv3`（kernel trace、PMC 计数器、ATT 指令级 trace） | §1.4 |
| shared memory | LDS（Local Data Share） | §2.0 |
| `__syncthreads()` | `s_barrier` | §2.0 |
| 全芯片统一的 L2 | XCD：8 个芯粒，各有独立 L2；其后是 256 MB MALL | §2.0 |

**路线图**

| 版本 | 优化 | 引入的 AMD 硬件概念 | 对应的 NVIDIA 经验 |
|---|---|---|---|
| v0 | 每 wave 算 128×128，寄存器复用 | CU / SIMD / wave64、MFMA 操作数排布、VGPR / AGPR 与 occupancy、`s_waitcnt` | `mma.sync` fragment、寄存器与 occupancy、SASS 依赖屏障 |
| v1 | LDS 共享 + XOR swizzle | LDS 容量与 bank、没有异步拷贝、`s_barrier`、XCD 与 L2 | smem tiling、`Swizzle<3,4,3>`、`__syncthreads` |
| v2 | 单 wave 软件流水 | 发射模型、ILP、`sched_group_barrier`；Hopper 方案为什么搬不过来 | 多级流水、warp specialization、TMA / WGMMA |

---

## 1. v0：最直观的实现

任务均分到 304 个 CU：每个 CTA 只有 1 个 wave，算 C 的 `[BM, BN]` 块，直接从显存读入寄存器，然后做 MFMA。SIMD 之间不协作。

### 1.0 硬件：CU、wave64 与寄存器

- **CU 与 SIMD**：MI300X 有 8 个 XCD × 38 = 304 个 CU。每个 CU 有 4 个 SIMD，每个 wave（64 lane）固定在一个 SIMD 上执行。一个 CTA 只会在 1 个 CU 上运行；4 wave/CTA 时正好每个 SIMD 一个 wave。
- **矩阵算力**：每个 SIMD 有一个 Matrix Core，BF16 512 FLOP/cycle。`v_mfma_f32_16x16x16_bf16` 一条 8192 FLOP，占 16 cycle。整卡峰值 304 × 4 × 512 FLOP × 2.1 GHz = **1307 TFLOPS**。
- **寄存器**：每个 lane 最多 512 个 32 位寄存器，分成 256 VGPR + 256 AGPR。AGPR 主要放 MFMA 累加器（gfx90a 起 MFMA 可以直接读写 AGPR）。另有每 wave 共享的 SGPR（≤ 102 个）放标量。每个 SIMD 的 VGPR+AGPR 共 128 KB，所以 wave 用的寄存器越多，一个 SIMD 能同时驻留的 wave 越少：用满 512 时只能驻留 1 个。

> **NVIDIA 对照**
>
> - CU ≈ SM，SIMD ≈ SM 子分区（各有一个矩阵单元）。但单个 CU 的矩阵算力（2048 FLOP/cycle）只有 H100 SM（4096）的一半，MI300X 靠 304 个 CU（H100 是 132 个 SM）取胜。
> - wave64 的 lane 数是 warp 的 2 倍，所有“哪个 lane 拿哪几个元素”的推导都要按 64 重做。
> - 每 lane 寄存器上限是 NVIDIA 的约 2 倍（512 vs 255），而且累加器有独立的 AGPR 文件。代价是寄存器数在编译期固定，没有 `setmaxnreg`。
> - 标量值（循环变量、基址）放 SGPR、由 SALU 执行，作用类似 NVIDIA 的 uniform datapath，但更显式：ISA 里 `s_` 开头的就是标量指令。

### 1.1 FlyDSL 基本结构

```python
A = fx.make_view(fx.get_iter(A), fx.make_ordered_layout((8, K//8, 16, BM//16, grid_m), 0))
A = A[None, None, None, None, bid_m]                 # None = 保留该维，其余维取定值
row, kg = tid % 16, tid // 16
for k, acc in range(fx.Index(0), fx.Index(K//32), fx.Index(1), init=acc_init):   # 运行时 scf.for
    kk = fx.Int32(k) * 4 + kg                        # 循环变量是 index 类型，tid 是 Int32
    a_tile = [A[None, kk, row, i].load().bitcast(fx.Int16) for i in range_constexpr(NI)]   # 16 B load
    ...
    results = yield acc                               # 累加器作为 loop-carried 值
```

- `make_ordered_layout(shape, 0)`：第 0 维最连续。把 `A[M, K]` 看成 `(8, K/8, 16, BM/16, grid_m)` 的 5 维张量，切片和下标就直接对应“哪个 lane 读哪 16 字节”，不需要手写地址计算。
- `range_constexpr` 是编译期展开；`range(fx.Index(...), init=...)` 是运行时循环。边界必须是 DSL 类型，写成 Python int 会被展开。

> **NVIDIA 对照**：思路和 CuTe 的 layout 代数相同，用 shape/stride 描述张量，用切片把“线程 → 数据”的映射写出来；`None` 对应 CuTe 的 `_`，`range_constexpr` 对应 CuTe DSL 的同名函数。

### 1.2 MFMA 操作数排布

`v_mfma_f32_16x16x16_bf16`：lane `l` 提供 A 第 `l%16` 行、K 组 `l//16` 的 4 个元素，B 同理；结果 lane `l` 持有 C 的 `rows 4*(l//16)+0..3, col l%16`。

技巧：每个 lane 一次 load **8 个连续 K（16 B，`global_load_dwordx4` / `ds_read_b128`）**，前 4 个和后 4 个分别喂两条 MFMA。A、B 用同样的 K 排列，`Σk A·B` 不变。所以一个 wave 每步覆盖 32 个 K，所有 load 都是 16 B 宽。

**单条 MFMA 的 lane 排布**（格子里是提供 / 持有该元素的 lane 号，B 与 A 相同，把“行”换成 N）：

```text
A 操作数：16 行(M) × 16 K                C 结果：16 行(M) × 16 列(N)
每格 = 1 个 lane 提供的 4 个 bf16          每格 = 1 个 lane 持有的 4 个 fp32（竖着 4 行）

        K 0-3  K 4-7  K 8-11 K 12-15               N 0   N 1   N 2  ...  N 15
row  0   L0     L16    L32    L48         M  0-3    L0    L1    L2   ...  L15
row  1   L1     L17    L33    L49         M  4-7    L16   L17   L18  ...  L31
row  2   L2     L18    L34    L50         M  8-11   L32   L33   L34  ...  L47
 ...                                      M 12-15   L48   L49   L50  ...  L63
row 15   L15    L31    L47    L63
```

**16 B load 的排布与访存合并**：把两条 MFMA 的 K 拼起来，一次 `global_load_dwordx4` 覆盖 16 行 × 32 K。A 在显存里按行存放，K 连续，行与行之间相隔 `K × 2 B = 16 KB`：

```text
        K 0-7   K 8-15  K 16-23  K 24-31        每格 16 B；MFMA #0 用每格前 4 个 K，MFMA #1 用后 4 个
row  0  [ L0     L16     L32      L48 ]  ← 同一行连续 64 B：这 4 个 lane 合并成 1 次访问
row  1  [ L1     L17     L33      L49 ]
 ...
row 15  [ L15    L31     L47      L63 ]
           ↑ 相邻 lane L0、L1、…、L15 落在 16 个不同的行，地址相隔 16 KB，不能合并
```

- 能合并的是 `{Lr, Lr+16, Lr+32, Lr+48}`：同一行连续的 64 B，落在同一条 128 B cache line 的一半里。
- 一条指令要访问 16 行，也就是 16 条 cache line，每条只用一半；另一半要等下一个 K 步再读。
- 对照 v1 的搬运排布（§2.2，`st_c = tid % 8`）：8 个**相邻** lane 读同一行的 128 B，一条指令只访问 8 条 cache line，而且每条都用满：

```text
        K 0-7  K 8-15  ...  K 56-63
row 0  [ L0     L1     ...   L7  ]  ← 1 条完整的 128 B cache line
row 1  [ L8     L9     ...   L15 ]
 ...
row 7  [ L56    L57    ...   L63 ]
```

所以 v0 直接按 MFMA 排布从显存读，合并效率只有一半；v1 先按合并友好的排布读到寄存器、写进 LDS，再按 MFMA 排布从 LDS 读出来。

> **NVIDIA 对照**：相当于 `mma.sync.m16n8k16` 的 fragment 排布，只是一条指令覆盖 16×16×16、由 64 个 lane 一起提供操作数。NVIDIA 上常用 `ldmatrix` 把 smem 装成 fragment；CDNA3 没有 `ldmatrix`，直接按上面的排布用 16 B 的 load 读。

### 1.3 优化原理：寄存器复用

一个 wave 算 `BM×BN`，每个 K 读 `BM+BN` 个元素、做 `2·BM·BN` 次 FLOP：寄存器层的算术强度 `I = BM·BN/(BM+BN)` flop/B（bf16）。

| wave tile | I (flop/B) | 累加器 / lane | 结果 | 时间 (us) | TFLOPS |
|---|---|---|---|---|---|
| 64×64 | 32 | 64 | | 8051 | 162 |
| 128×128 | 64 | 256（AGPR 用满） | A/B 各 32 VGPR | 3863 | **338** |
| 128×256 | 85 | 512 | spill | | ~90 |

强度从 32 翻到 64，性能也约翻倍（2.08×），说明 v0 受访存限制。128×256 的强度虽然更高，但累加器放不下，spill 到 scratch（显存）后反而慢到约 90 TFLOPS。

在不 spill 的前提下把 BM、BN 加到最大（128×128）。副作用是 occupancy 降到 1 wave/SIMD，没法靠 wave 切换（TLP）来掩盖延迟。

> **NVIDIA 对照**：在 NVIDIA 上，一个 warp 算 128×128 需要每线程 512 个 fp32 累加寄存器，超过 255 的上限，所以通常由多个 warp（Hopper 上是一个 warpgroup）分担。CDNA3 的 wave64 + AGPR 让一个 wave 就能扛住 128×128。occupancy = 1 在这里是常态，和 NVIDIA 上高性能 GEMM 的思路一致：用寄存器换复用，延迟靠 ILP 而不是靠切 warp。


### 1.4 观察硬件行为

```bash
FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python3 v0.py   # IR + 最终 ISA
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 rocprofv3 -i trace.yaml -- python3 v0.py       # ATT thread trace
rocprofv3 --stats --kernel-trace -f csv -- python3 v0.py                         # 基本信息
```

- ISA：每步 `global_load_dwordx4` 之后紧跟 `s_waitcnt vmcnt(0)` 才开始 MFMA，访存和计算串行。
  - `s_waitcnt`：每个 wave 有按发射顺序计数的计数器，`vmcnt` 数在途的显存访问，`lgkmcnt` 数在途的 LDS / 标量访问。`vmcnt(N)` 表示“等到在途的只剩 N 条”，也就是等最老的那些完成；由编译器根据数据依赖插入。`vmcnt(0)` 就是全部等完。
- ATT `occupancy.json`：每个 CU 的 4 个 wave（每 SIMD 一个）在开始时同时上线、到结束才下线，中间没有新 wave 可以切换。
- 问题：同一 CU 的 4 个 wave 各读各的 A、B 行，没有共享。

> **NVIDIA 对照**
>
> - ISA dump ≈ `cuobjdump -sass`；`rocprofv3 --pmc` ≈ Nsight Compute 的硬件计数器；ATT 能看到每条指令的发射时刻和 stall，Nsight 里最接近的是 Source 页的 warp stall 采样，但 ATT 是完整 trace 而不是采样。
> - `s_waitcnt` 对应 SASS 控制位里的依赖屏障（SB0~5）。区别是 AMD 只能“等最老的若干条”，不能指定等哪一条，所以**发射顺序决定了流水能有多深**（§3.7）。

---

## 2. v1：引入 LDS

CDNA3 的 4 个 SIMD 共享一个 LDS，所以 4 wave/CTA 可以通过 LDS + barrier 协作加载 A、B，再各自从 LDS 读取复用。目的和 v0 加大 wave tile 一样，是提高数据复用；LDS 的作用是把复用从 wave 级扩大到 CTA 级，进一步降低外存访问。

### 2.0 硬件：LDS、barrier 与缓存层次

- **LDS**：每 CU 64 KB，4 个 SIMD 共享。32 bank × 4 B，带宽 128 B/clk。
- **搬运路径**：CDNA3 的 global→LDS 直通只有 4 B/lane（`buffer_load_dword ... lds`），太窄。实用的路径是 global → VGPR（`global_load_dwordx4`）→ `ds_write_b128` → LDS，每个字节都要经过寄存器。
- **同步**：CTA 内只有 `s_barrier`，所有 wave 一起到达才放行。等本 wave 的 LDS 读写完成用 `s_waitcnt lgkmcnt(N)`。
- **缓存**：L1 32 KB/CU；L2 4 MB/XCD，8 个 XCD 之间不共享；其后是 256 MB MALL（Infinity Cache）和 HBM（MI300X 5.3 TB/s）。CTA 按编号轮流分配到 8 个 XCD。

> **NVIDIA 对照**
>
> - LDS ≈ smem，但只有 H100（228 KB）的约 1/4。NVIDIA 上常见的 3~7 级 smem 环形缓冲在这里放不下（§3.2）。
> - 没有 `cp.async` / TMA：搬运要占 VGPR 和发射槽，写法更像 Ampere 之前的 `ld.global` → 寄存器 → `st.shared`。
> - `s_barrier` ≈ `__syncthreads()`；没有命名 barrier（`bar.sync id, n`），也没有 mbarrier。
> - bank 结构相同（32 × 4 B），CUTLASS 的 XOR swizzle 思路可以直接用。
> - L2 不是全芯片统一的：两个 CTA 读同一块数据，只有落在同一个 XCD 上才能在 L2 命中（§5）。

### 2.1 切分

- 64 KB LDS 尽量占满：CTA 算 256×256，4 个 wave 2×2 均分，每个 wave 仍算 128×128（保持寄存器复用）。
- 每个 K 步 `BK=64`：A、B 各 256×64 bf16 = 32 KB。
- L2 层强度：每 K 步读 64 KB、做 `2·256·256·64` FLOP → 128 flop/B，是 v0 的 2 倍。
- LDS 层：每 K 步读 128 KB（每 wave 32 KB）+ 写 64 KB，约 48 B/clk/CU，是 LDS 带宽 128 B/clk 的 37%。

```text
                              B^T : K x N（B 在显存里是 [N, K]、K 连续，这里转置画）
                              +----------------+----------------+
                              |                |                |
                              +================+================+   <- 当前 K 步，BK = 64
                              |######## B tile: 64 x 256 #######|
                              |########### (LDS, 32 KB) ########|
                              +================+================+
                              |                |                |
                              +----------------+----------------+
                              |<---- 128 ----->|<---- 128 ----->|
                                    wn = 0           wn = 1
   A : M x K
   +-------+=======+-------+  +----------------+----------------+
   |       |#######|       |  |                |                |  ^
   |       |#  A  #|       |  |    wave 0      |    wave 2      |  |
   |       |# tile#|       |  |   128 x 128    |   128 x 128    | 128   wm = 0
   |       |#     #|       |  |  acc in AGPR   |  acc in AGPR   |  |
   |       |#256 x#|       |  |                |                |  v
   |       |#  64 #|       |  +----------------+----------------+
   |       |#     #|       |  |                |                |  ^
   |       |# LDS #|       |  |    wave 1      |    wave 3      |  |
   |       |#32 KB#|       |  |   128 x 128    |   128 x 128    | 128   wm = 1
   |       |#     #|       |  |  acc in AGPR   |  acc in AGPR   |  |
   |       |#######|       |  |                |                |  v
   +-------+=======+-------+  +----------------+----------------+
           |<-64-->|                C tile : 256 x 256（一个 CTA）
            BK = 64
```

- `#` 区域是当前 K 步放进 LDS 的 A、B tile，K 方向每步前进 64。
- C tile 按 `wave = tid // 64`、`wm = wave % 2`、`wn = wave // 2` 分成 2×2。每个 wave 读 A 的一半（`wm`）和 B 的一半（`wn`）。A 的上半被 wave 0、2 共用，B 的左半被 wave 0、1 共用，这就是 CTA 级复用。

| 数据 | 尺寸 | 放在哪 | 每 lane 寄存器 |
|---|---|---|---|
| C 累加器 | 每 wave 128×128 fp32 | AGPR，整个 K 循环常驻 | 256 |
| A、B tile（当前 K 步） | 各 256×64 bf16 = 32 KB | LDS（共 64 KB，4 个 wave 共享） | — |
| 搬运暂存（global → LDS 中转） | A、B 各 256×64，由 256 个线程分担 | VGPR | A、B 各 8 × 16 B = 各 32 |
| MFMA 操作数 | 每 wave 每次取 32 个 K：A 128×32、B 128×32，每个 K 步取 2 次 | VGPR | A、B 各 8 × 16 B = 各 32 |

### 2.2 每个 K 步的三段

```python
a_st = [A[None, st_c, kt, st_r, p].load() for p in range_constexpr(BM // 32)]   # ① global -> 暂存寄存器
gpu.barrier()                                                                   # 所有 wave 读完上一块
for p in range_constexpr(BM // 32):
    lds_a[None, st_c_sw, p * 32 + st_r].store(a_st[p])                         # ② 暂存寄存器 -> LDS
gpu.barrier()                                                                   # 本块对所有 wave 可见
a_tile = [lds_a[None, ck, wm * WM + i * 16 + row].load() ...]                    # ③ LDS -> 寄存器 -> MFMA
```

- **合并访存**：`st_c = tid % 8, st_r = tid // 8`，8 个相邻线程读一行的 128 B，一个 wave 一条指令读 8 行 = 1024 B 连续。
- **申请 LDS**：`@fx.struct` 声明 `fx.Array[fx.BFloat16, n, 16]`，`fx.SharedAllocator().allocate(...)`，再 `.view(layout)` 得到可切片的 LDS 张量。
- **FlyDSL 内置 swizzle**：手写 XOR 是最直接的思路，但不一定最自然。FlyDSL 也提供了一层布局表达：

```python
swz = fx.static(fx.SwizzleType.get(3, 3, 3))
layout = fx.make_composed_layout(swz, fx.make_ordered_layout((8, 8, BM), 0))

# 逻辑坐标 (chunk, row) -> 物理偏移
ptr = fx.add_offset(lds.a.ptr, fx.crd2idx((0, c, r), layout))
view = fx.make_view(ptr, fx.make_layout(8, 1))
```

这和手工写 `chunk ^ (row % 8)` 是同一个映射：把 `row % 8` 这一位组拿去 XOR 到 `chunk` 的低位，保证同一拍访问的 8 行不再落到同一组 bank。FlyDSL 不直接支持对 `composed layout` 做向量 `load/store`（这个版本里会报 `MemRefLoadVecOp: MemRefType with ComposedLayout is not supported`），所以通常做法是：让它只算地址，拿到 `crd2idx` 的 swizzled offset，再把这 8 个元素包成一个普通的 8 元素 view，继续做 `load()/store()`。这正是 `v1.py` 中 `SWIZZLE="fly"` 的实现方式：相同语义，保留手写版可读性与更低的 VGPR 成本。

### 2.3 LDS 硬件结构与 bank conflict

参考 AMD 官方文档：<https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/cdna/pipeline-descriptions.html#local-data-share-lds>

LDS 不是“一个大数组”这么简单，它更像一个 4 个 SIMD 共享的高带宽共享存储器网络。对 CDNA3 来说，理解它的关键是：同一个 CU 的多个 wave / SIMD 会并发读写 LDS，而 bank 访问冲突会在非常低的层面上发生，因此 swizzle 不是抽象技巧，而是把“逻辑地址”重新映射成“物理 bank 分布”的硬件优化。

```text
SIMD / wave  lane ──> vector op decode ──> request selector ──> address crossbar
                                              │
                                              ├──> 32-bank LDS memory array
                                              │          (32 x 4 B, 64 KB per CU)
                                              └──> read data crossbar ──> SIMD

                         其中：write path / atomic path / conflict detection / bank scheduler
                               也在同一层次中参与仲裁，保证多个 wave 的请求按 bank 维度排队。
```

简化后的理解：

- 每个 LDS 访问都要先 decode 成“哪个 SIMD、哪个 lane、哪个地址、是否读/写/原子”。
- 地址会经过一个 crossbar，把不同 lane 的逻辑地址映射到 bank 物理组中。
- 32 个 bank 以 4 B 为粒度组织，bank scheduler 会在同一周期里决定哪些请求可同时命中。
- 若多个 lane 试图同时命中同一 bank 组，冲突就会发生；冲突不是纯“编程错误”，而是相同 bank 地址同时被多个线程访问时，硬件必须把请求串行化。
- 读返回的数据再经过一个 read-data crossbar 送回相应 lane；写入、原子和同步也都经过同一套共享介质。

这也是为什么 v1 里 `LDS` 的布局设计那么重要：如果逻辑地址和 bank 映射不对，`ds_read_b128`/`ds_write_b128` 会在同一拍上把 8 个 lane 都打到同一组 bank，导致 `SQ_LDS_BANK_CONFLICT` 上升，吞吐明显掉。

#### 为什么会 bank conflict？

对 64 KB 的 LDS 来说，一行 64 个 bf16 = 128 B = 刚好 32 个 bank。一个 `ds_read_b128` 访问会在一个拍内同时读 8 个 lane 的 128 B 数据，通常是“8 行 × 同一 chunk”这种模式。

如果我们按最自然的逻辑布局存：

```text
row 0: chunk 0,1,2,3,4,5,6,7
row 1: chunk 0,1,2,3,4,5,6,7
row 2: chunk 0,1,2,3,4,5,6,7
...
```

那么同一拍里，8 个 lane 读的是 `row = 0..7, chunk = c`，它们都会落到同一组 bank（因为 bank 的低位和 chunk 相关、而 row 仅决定高位偏移/跨组排列）。这就是 8 路冲突：本来应该 8 个 bank 组并发拿到 8 条数据，但在物理上都挤在同一组 bank 上。

#### XOR swizzle 是怎么修的？

XOR swizzle 的本质是“对物理地址做一个 bit 翻转”，而不是“用更大的 padding”。对于一个 `16 B chunk`，逻辑地址可以看成：

```text
offset = e + 8 * chunk + 64 * row
```

其中：

- `e`：当前 chunk 内 0..7 的元素偏移
- `chunk`：每行里 8 个 16 B chunk 的编号
- `row`：当前行号（0..7，或更大）

如果把 `row % 8` 这部分 bit XOR 到 `chunk` 的低位，就变成：

```text
physical_chunk = chunk ^ (row % 8)
```

这意味着在同一拍访问 `row=0..7, chunk=c` 时，8 个 row 会对应到 8 个不同的 `physical_chunk`，于是 8 个访问分散到不同 bank 组；同一行写 8 个 chunk 时也一样，写入分散了，不再集中在相同 bank。此时 `ds_read_b128` 与 `ds_write_b128` 都能保持低冲突，`SQ_LDS_BANK_CONFLICT ≈ 0`。

- **XOR swizzle 不占额外空间**：它只是改变“逻辑地址 → 物理地址”的映射，和 padding 不同。
- **padding 会浪费空间**：把一个 bank 不对齐的布局强行补齐后，64 KB 的有效数据量会下降。
- **FlyDSL 的 `SwizzleType.get(3, 3, 3)` 做的正是做相同的翻转**，只不过它把这个映射放进 `composed layout`，把“地址映射”的语义表达给抽象掉了。

这也是为什么在 CDNA3 上，swizzle 是一个工程常识：不是为了“多一个优化器开关”，而是为了让 bank 访问维持在 0 冲突，保持 LDS 带宽稳定地接近 128 B/clk 这一硬件上限。

### 2.4 v1 的问题

沿用 §1.3 的指标对比；v1 的 CTA tile 是 256×256，由 4 个 wave 各算 128×128，因此寄存器层强度和每 lane 累加器数量仍按 **wave tile** 计算。时间和吞吐取自同一次 8192×9728×8192 BF16 GEMM 测量。

| wave tile | I (flop/B) | 累加器 / lane | 结果 | 时间 (us) | TFLOPS |
|---|---|---|---|---|---|
| 64×64（v0） | 32 | 64 | | 8051 | 162 |
| 128×128（v0） | 64 | 256（AGPR 用满） | A/B 各 32 VGPR | 3863 | **338** |
| 128×256（v0） | 85 | 512 | spill | | ~90 |
| 128×128（v1；CTA 256×256） | 64 | 256（AGPR 用满） | A/B 经 LDS 跨 wave 共享 | 2965 | **440** |

此时还没有软件流水：每个 K 步严格串行，发 global load → 等 `vmcnt(0)` → barrier → 写 LDS → barrier → 读 LDS + MFMA。1 wave/SIMD 下没有别的 wave 填空档。v1 相比 v0 128×128 快约 1.30 倍；L2 之外的读取从 5.59 GB 降到 1.24 GB（约为原来的 22%），但 MFMA 大部分时间仍在等待访存。

**如何测 L2 之外的实际读取量**：在 MI300X 上用 `rocprofv3 --pmc` 收集 `TCC_EA0_RDREQ_sum` 和 `TCC_EA0_RDREQ_32B_sum`（本仓库的 [采集脚本](slides/measure.sh) 会自动完成，并把结果写入 [计数结果](slides/data/counters.txt)）。只跑 v1 可在 `kernels/gemm` 目录执行：

```bash
rocprofv3 --pmc TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum \
    --kernel-include-regex 'gemm_v' -d /tmp/v1-pmc -o pmc -f csv \
    -- python3 bench.py v1
```

在输出的 `*counter_collection.csv` 中按 `Dispatch_Id` 汇总每个计数器的 `Counter_Value`；每次 dispatch 的读取字节数为 `(TCC_EA0_RDREQ_sum - TCC_EA0_RDREQ_32B_sum) × 64 + TCC_EA0_RDREQ_32B_sum × 32`。多次 dispatch 取均值、除以 `1e9` 得 GB，得到上面的 1.24 GB（v0 128×128 同法得到 5.59 GB）。**口径是 L2/TCC 向外发出的读请求字节数，包含后续 MALL 命中，不能当作 HBM 物理读取字节数**；不要把它与源代码中 A/B 张量大小或 L2 命中率直接混用。

---

## 3. v2：重叠搬运和计算 —— 单 wave 软件流水

前两步为了最大数据复用、降低外存访问，把 occupancy 做到了 1，没有 TLP 可用了。这时只能用 **ILP**：在同一个指令流里并行发出彼此无关的 vmem / MFMA / ds / valu 指令。要做到这一点，需要软件流水。

**延迟怎么藏：TLP（切 wave）与 ILP**

一次 global load 要几百到上千 cycle 才回来，一条 MFMA 只有 16 cycle。要让 MFMA 单元一直有活干，等数据的这段时间必须有别的指令可发。办法有两种：

```text
① 不藏：1 个 wave，代码是“load → 等 → 算”（v0 现状，§1.4 的 ISA 里 load 后紧跟 s_waitcnt vmcnt(0)）

wave A  : [LD 0]━━━━━━━ 等数据 ━━━━━━━[MFMA 0 ×4][LD 1]━━━━━━━ 等数据 ━━━━━━━[MFMA 1 ×4]
MFMA 单元:                              ████                                   ████     ← 大部分时间空闲

② TLP（切 wave）：同一 SIMD 上驻留多个 wave，每个 wave 的代码仍是①；
   A 卡在 s_waitcnt 时，调度器转去发 B、C 的指令

wave A  : [LD 0]━━━━━━━ 等数据 ━━━━━━━[MFMA ×4][LD 1]━━━━━━━ 等 ━━━━━━━ ...
wave B  :        [LD 0]━━━━━━━ 等数据 ━━━━━━━[MFMA ×4][LD 1]━━━━━━━ ...
wave C  :               [LD 0]━━━━━━━ 等数据 ━━━━━━━[MFMA ×4][LD 1]━━━ ...
MFMA 单元:                              ████████████  ...                       ← 靠 wave 数量填满

③ ILP：只有 1 个 wave，但代码里提前发出下一块的 load，再算当前块；
   load 在途期间同一个 wave 继续发 MFMA

wave A  : [LD 1][MFMA 0 ×4][LD 2][MFMA 1 ×4][LD 3][MFMA 2 ×4] ...
                ╰─ LD 1 在途 ─╯   ╰─ LD 2 在途 ─╯
MFMA 单元:       ████████████████████████████████████            ← 靠指令顺序填满
```

| | TLP（切 wave） | ILP（单 wave 内交错） |
|---|---|---|
| 谁来藏延迟 | 硬件调度器：一个 wave 卡住时发另一个 wave 的指令。所有驻留 wave 的寄存器一直在寄存器堆里，“切换”是零开销的逐拍选择，不是上下文切换 | 编译器 / 程序员：在需要等待之前，先发出与它无关的指令 |
| 代价 | 每个 wave 都要一份寄存器，wave 越多每个 wave 的寄存器越少，tile 只能变小、复用变差 | 要额外寄存器暂存在途的数据；指令顺序要精心安排（软件流水、`sched_group_barrier`） |
| 本例 | 128×128 tile 用满 512 个寄存器，每 SIMD 只能驻留 1 个 wave，**TLP 不可用** | v2 的做法（§3）：提前 2 个 K 步发 load，并把访存与 MFMA 交错 |

一句话：TLP 用“更多 wave”换延迟，ILP 用“更多寄存器 + 更好的指令顺序”换延迟。寄存器既要放大 tile 提高复用，又要支撑多个 wave 或在途数据，三者互相争抢；本教程选择把寄存器优先给大 tile，所以只能走 ILP。

设计完全仿照 [gfx942/intra_wave](https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/gfx942/intra_wave)，代码从 v1 修改而来。

### 3.0 硬件：发射模型，以及 Hopper 的做法为什么搬不过来

**发射模型**

- 每个 CU 的 4 个 SIMD 轮流发射，每个 SIMD 每 4 cycle 轮到一次。一次可以从不同 wave 发出最多 5 条不同类型的指令（VALU、SALU、VMEM、LDS、分支等），但**同一个 wave 一次只能发 1 条**。
- 1 wave/SIMD 时，一个 wave 每 4 cycle 最多发 1 条指令。一条 MFMA 占 16 cycle，所以两条背靠背的 MFMA 之间最多插 3 条别的指令，所有搬运都要挤进这些空隙（§3.7）。
- MFMA 发出后在矩阵单元里执行，后面不相关的指令可以继续发，依赖由硬件互锁处理。所以“异步”靠的是同一指令流里的交错，而不是专门的异步指令。

> **NVIDIA 对照**：Hopper 子分区调度器每 cycle 发 1 条，warp 多时靠切换 warp 藏延迟。CDNA3 上 1 wave/SIMD 没有 warp 可切换，只能靠 ILP，编译器（和 `sched_group_barrier`）排出来的指令顺序就是全部。

**既不是TLP也不是ILP, Hopper 上怎么做**

同样是“让搬运和 MMA 重叠”，Hopper 的标准答案是 warp specialization：一个 producer warpgroup 用 TMA 把数据搬进多级 smem，两个 consumer warpgroup 用 WGMMA 直接从 smem 算，双方用 mbarrier 按 stage 握手，并用 `setmaxnreg` 把 producer 的寄存器让给 consumer。结构对应 CUTLASS 3.x 的 `KernelTmaWarpSpecializedCooperative`（单 CTA，不含 cluster / multicast）：

<details>
<summary>Hopper warp-specialized TMA GEMM 伪代码</summary>

```c++
// ============================================================
// Hopper Warp-Specialized TMA GEMM 伪代码
// A[M,K] * B[K,N] -> D[M,N]，BF16 输入，FP32 累加
// CTA = 3 个 warpgroup（384 线程）：WG0 = producer，WG1/WG2 = consumer
// CTA tile 256x128：两个 consumer WG 沿 M 各算 128x128（cooperative）
// smem：N_STAGES 级 A/B 环形缓冲（H100 每 SM 228 KB，常见 4~7 级）
// ============================================================

__global__ void hopper_ws_tma_gemm(const __grid_constant__ CUtensorMap descA,   // TMA 描述符由 host 创建，
                                   const __grid_constant__ CUtensorMap descB,   // 必须以 __grid_constant__ 传入
                                   ...) {
    // ---------- 关键点 1：线程角色划分 ----------
    int  wg_id       = threadIdx.x / 128;          // 0 = producer, 1..2 = consumer
    int  warp_in_wg  = (threadIdx.x / 32) % 4;
    bool is_producer = (wg_id == 0);

    // ---------- 关键点 2：smem 环形缓冲 + mbarrier ----------
    extern __shared__ char smem[];
    Tile* sA[N_STAGES], *sB[N_STAGES];             // 指向 smem 内的各级 buffer
    __shared__ uint64_t full_barrier[N_STAGES];    // mbarrier 必须放在 smem
    __shared__ uint64_t empty_barrier[N_STAGES];   // full：数据已到；empty：可被覆盖

    if (threadIdx.x == 0) {
        for (int i = 0; i < N_STAGES; i++) {
            mbarrier_init(full_barrier[i], 1);               // 1 次 arrive（producer）+ tx 字节计数
            mbarrier_init(empty_barrier[i], NUM_CONSUMER_WG); // 每个 consumer WG 1 次 arrive
        }
        fence_barrier_init();                       // 让 async proxy（TMA 单元）看到初始化结果
    }
    __syncthreads();

    // ---------- 关键点 3：寄存器重新分配（setmaxnreg） ----------
    // 以每线程 168 个寄存器启动（384 × 168 = 64512），再按角色重新分配：
    // 128 × 24 + 256 × 240 = 64512 ≤ 65536（每 SM 64K 个 32 位寄存器）
    if (is_producer) warpgroup_reg_dealloc<24>();
    else             warpgroup_reg_alloc<240>();

    // ============================================================
    // producer：整个 WG 只有 1 个线程发 TMA
    // ============================================================
    if (is_producer) {
        // elect_one_sync() 是“每个 warp 选 1 个线程”，所以还要限定 warp 0，
        // 否则 4 个 warp 各发一次 TMA、各 arrive 一次，full_barrier 计数就错了
        if (warp_in_wg == 0 && elect_one_sync()) {
            int stage = 0, phase = 1;               // producer 从 phase=1 开始：首轮 wait 直接通过（buffer 初始为空）
            for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                mbarrier_wait(empty_barrier[stage], phase);                       // 等 consumer 释放该级
                mbarrier_arrive_expect_tx(full_barrier[stage], BYTES_A + BYTES_B); // 声明将到达的字节数
                tma_load_2d(descA, sA[stage], {k_tile * BK, m0}, full_barrier[stage]); // 异步；数据写入 smem 后
                tma_load_2d(descB, sB[stage], {k_tile * BK, n0}, full_barrier[stage]); // 硬件按字节数递减 tx
                if (++stage == N_STAGES) { stage = 0; phase ^= 1; }
            }
        }
    }
    // ============================================================
    // consumer：2 个 WG，各自用 WGMMA 算 128x128
    // ============================================================
    else {
        float acc[128] = {0};                      // 累加器在寄存器：128x128 fp32 / 128 线程
        int m_off = (wg_id - 1) * 128;             // cooperative：沿 M 切半（ping-pong 则两个 WG 各算不同 tile）
        int stage = 0, phase = 0, prev = -1;
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            mbarrier_wait(full_barrier[stage], phase);   // producer 已 arrive 且 tx 归零 = TMA 数据已写入 smem

            // ---------- 关键点 4：WGMMA（异步，4 个 warp 共同发射） ----------
            warpgroup_arrive();                           // wgmma.fence：保证此前对累加器寄存器的读写有序
            wgmma_async(acc, sA[stage] + m_off, sB[stage]); // B 必须来自 smem；A 可来自 smem 或寄存器
            wgmma_commit_group();
            wgmma_wait<1>();                              // 只留本次在飞，上一级的 wgmma 已读完 smem
            if (prev >= 0 && threadIdx.x % 128 == 0)
                mbarrier_arrive(empty_barrier[prev]);     // 释放上一级给 producer
            prev = stage;
            if (++stage == N_STAGES) { stage = 0; phase ^= 1; }
        }
        wgmma_wait<0>();
        if (threadIdx.x % 128 == 0) mbarrier_arrive(empty_barrier[prev]);

        // ---------- 关键点 5：epilogue ----------
        store_accumulator(acc);   // 寄存器 -> smem -> TMA store，或直接 st.global
    }
}
```

</details>

**这些关键技术在 CDNA3 / CDNA4 上的情况**

| Hopper 技术 | 作用 | CDNA3（gfx942，MI300X/MI325X） | CDNA4（gfx950，MI350X/MI355X） | 本教程的替代做法 |
|---|---|---|---|---|
| **TMA**（张量描述符，单线程发起，硬件生成地址、越界填充、smem swizzle） | 搬运不占寄存器，几乎不占发射槽 | **无**。每个 lane 自己算地址；global→LDS 直通只有 4 B/lane（`buffer_load_dword ... lds`） | **无 TMA**。新增 16 B/lane 的 `global/buffer_load_lds_dwordx4`，数据不经 VGPR，但仍是每 lane 一条地址、用 `vmcnt` 计数；LDS 目标地址由 `M0` + lane 线性决定，swizzle 只能在 global 地址一侧做 | global → VGPR → `ds_write`（v1/v2）；buffer resource 的 `num_records` 越界读返回 0，部分代替 TMA 的边界处理；XOR swizzle 手算 |
| **mbarrier + 事务计数**（arrive / wait 分离、phase 翻转、TMA 完成自动递减） | 生产者与消费者按 stage 精确握手，不需要全 CTA 同步 | **无**。只有 `s_barrier`（CTA 内**所有** wave 一起）和每个 wave 自己的 `s_waitcnt vmcnt/lgkmcnt` | 同 CDNA3 | 用全 CTA `s_barrier` + 编译器插入的 `s_waitcnt`；自己用 LDS 标志位 + 轮询（`s_sleep`）可以模拟，但开销大 |
| **Warp specialization**（producer / consumer WG） | 搬运和计算由不同 warp 负责，靠 warp 调度器重叠 | 可以写，但**不划算**：没有部分 barrier；producer wave 要和某个 consumer 共用一个 SIMD 的发射槽和寄存器 | 同 CDNA3 | 单 wave 软件流水（v2，ILP）；或每 SIMD 2 个 wave 用 `s_barrier` + `s_setprio` 乒乓（§5 inter_wave） |
| **setmaxnreg**（运行时按 WG 调整寄存器数） | producer 让出寄存器给 consumer 放大累加器 | **无**。同一 kernel 所有 wave 的 VGPR/AGPR 数在编译期固定 | 同 CDNA3 | 1 wave/SIMD 下每 wave 512 个（256 VGPR + 256 AGPR）都给计算；如果再加一个 producer wave，每 wave 最多只剩一半，累加器 tile 就得减半 |
| **WGMMA**（异步、操作数直接读 smem、`commit/wait_group`） | 不用先把 A/B 搬进寄存器；MMA 在后台跑 | **无**。MFMA 操作数只能来自 VGPR/AGPR，每个 K 步都要先 `ds_read`；没有 `wait_group` | 同 CDNA3；MFMA 吞吐翻倍（`v_mfma_f32_16x16x32_bf16`），新增 `ds_read_b64_tr_b16`（读 LDS 时转置） | `ds_read` 与 MFMA 在同一指令流里穿插（`sched_group_barrier`，§3.7） |
| **大 smem 多级流水** | 3~7 级环形缓冲藏住 HBM 延迟 | LDS 64 KB/CU，256×256×64 只能放 1 级 | LDS 160 KB/CU，可以 2 级 | 用寄存器当第二级 buffer（§3.2、§3.6） |
| **Thread Block Cluster / DSMEM / TMA multicast** | 相邻 CTA 共享一次 global 读 | **无** | **无** | 依赖 L2 / MALL 复用；XCD 感知的 CTA 重映射（§5） |
| **TMA store**（epilogue） | 寄存器 → smem → 一条指令批量写回 | **无** | **无** | 每个 lane 自己 `global_store` / `buffer_store` |

结论：

- **CDNA3** 上，搬运必须经过 VGPR，同步只有全 CTA `s_barrier` 和每 wave 的计数器，矩阵指令只能读寄存器。Hopper 那一套“TMA + mbarrier + WGMMA + setmaxnreg”撑起来的 warp specialization 一样都没有，所以 v2 走单 wave 软件流水，把搬运、`ds_read`、`ds_write` 都排进同一个指令流里和 MFMA 穿插。
- **CDNA4** 补上了 16 B 的 global→LDS 直通和 160 KB LDS，能力上接近 Ampere 的 `cp.async` + 多级 smem 流水（不占 VGPR、可以双缓冲），但仍没有 TMA、mbarrier、WGMMA、setmaxnreg、cluster，同步模型和 CDNA3 一样。

### 3.1 要藏哪些延迟

| 延迟来源 | 量级 | 藏法 |
|---|---|---|
| global load（L2/MALL/HBM） | 数百~上千 cycle | 提前 2 个 K 步发出（GR k+2） |
| 寄存器 → LDS | 等 global 数据到达 | 提前 1 个 K 步写入（LW k+1） |
| LDS → 寄存器 | ~100 cycle | 提前 1 个 region（~1024 cycle）读出（LR） |

### 3.2 为什么不是 LDS 双缓冲

- 256×256×64 一级就是 64 KB，LDS 正好放一级。
- 把 BK 降到 32 做双缓冲（8 KB × 8 = 64 KB）：每行只有 64 B，一条 `global_load_dwordx4` 要碰 16 条 cache line，TCP 处理时间加倍。参考实现实测 400 vs 476 TFLOPS，更慢。
- 结论：**保持 BK=64，用寄存器当第二级 buffer**，LDS 原地刷新。NVIDIA 上对应的做法是 TMA/cp.async + 多级 smem 环形缓冲，在 CDNA3 上做不到。

### 3.3 半 tile 切分与 4 个 region

A 沿 M 切成 `A_top / A_bot`，B 沿 N 切成 `B_left / B_right`，每个 16 KB，占一个 LDS slot（4 个 slot 正好 64 KB）。`(A 半, B 半)` 组成 C 的一个 128×128 象限；每个 wave 在每个象限各算 64×64。

**先看动画，再对照下面的表和 §3.4–3.5 的代码**：动画只播放稳态循环体（R0 → R1 → barrier X → R2 → R3 → barrier Y），不重复 prologue。每一步的操作同时开始、在 2 秒内完成，随后停顿 3 秒再进入下一步；底部进度条仅在这 2 秒内前进，并显示当前步骤（共 6 步，包含两个 barrier）。这些秒数只是演示节奏，**不是硬件执行延迟**。开头已预备好 A_top/B_left(k) 的 MFMA 操作数、A_bot/B_right(k) 的 LDS 数据以及 k+1 的 GR 暂存；结尾回到相同的存储分布，只把 k 整体推进一步。按典型 GEMM 布局，A 从左下 HBM 向右经 GR 暂存 VGPR、LDS、MFMA 操作数 VGPR；Bᵀ 从右上 HBM 向下经过同样四层。计算 `C_tl += A_top(k) × B_left(k)` 时，**A/B 留在操作数 VGPR 原位**，两条流动箭头表示 MFMA 从中读取；箭头抵达后 `C_tl` 加亮。R1–R3 依次点亮另外三个象限，**亮起后保持至本轮结束，下轮重新全暗**。这只是数据依赖示意，真实 MFMA 不会把操作数搬进 C，而是在 AGPR 中更新累加器。蓝色、金色、青色外框分别代表 HBM、LDS、VGPR（虚线青框是 GR 暂存）；空框表示本次取用，不代表 HBM 数据被删除。同一 region 的访存和 DOT 实际交错发射。

![v2 软件流水：A/B 半 tile 在外存、预取暂存 VGPR、LDS、MFMA 操作数 VGPR 间流动并计算 C](slides/figs/v2_pipeline.svg)

[单独打开动画（SVG）](slides/figs/v2_pipeline.svg)

一个 K 步 = 4 个 region，每个 region = 一个象限的 64 条 MFMA（约 1024 cycle），再穿插三种访存：

| region | DOT | LR（LDS→寄存器） | LW（寄存器→LDS） | GR（global→寄存器） |
|---|---|---|---|---|
| 0 | `C_tl = A_top·B_left` | `A_bot(k)` | `B_left(k+1)` | `B_left(k+2)` |
| 1 | `C_bl = A_bot·B_left` | `B_right(k)` | `A_top(k+1)` | `A_top(k+2)` |
| — | **barrier X** | | | |
| 2 | `C_tr = A_top·B_right` | `B_left(k+1)` | `A_bot(k+1)` | `A_bot(k+2)` |
| 3 | `C_br = A_bot·B_right` | `A_top(k+1)` | `B_right(k+1)` | `B_right(k+2)` |
| — | **barrier Y** | | | |

- LR 读的总是**下一个** region 要用的操作数，所以 LDS 延迟有一个 region 的时间来藏。
- 相邻 region 共享一个操作数：`b_left` 用于 R0、R1，`a_top` 用于 R0、R2，依此类推。
- 某个 slot 读完以后，过一个 region 就重写；重写以后，再过一个 region 才被读。所以**单级 LDS 足够**。
- 流水深度：GR(k+2) → LW(k+1) 隔一整个 K 步（约 4096 cycle），用来藏 HBM/L2 延迟；LR(k+1) → DOT(k+1) 隔一个 region（约 1024 cycle），用来藏 LDS 延迟。这和 gfx950 双缓冲的深度相同。

### 3.4 正确性：为什么每 K 步只要 2 个 barrier

LDS 是 4 个 wave 共享的：一个 wave 写的数据要被其它 wave 读，所以同一 slot 的读和写之间必须隔一个 barrier。逐个 slot 检查：

| slot | 上次读（WAR） | 写 | 下次读（RAW） | 隔开它们的 barrier |
|---|---|---|---|---|
| B_left | 上一 K 步 R2 | R0 | R2 | Y（上一步末尾） / X |
| A_top | 上一 K 步 R3 | R1 | R3 | Y / X |
| A_bot | R0 | R2 | 下一 K 步 R0 | X / Y |
| B_right | R1 | R3 | 下一 K 步 R1 | X / Y |

每一对读写都恰好跨过 X 或 Y。prologue 末尾还需要一个 barrier：prologue 里读了 `b_left(0)`、`a_top(0)`，要等它们读完，R0 才能重写 B_left。

`gpu.barrier()` 会先等本 wave 的 LDS 操作完成（`s_waitcnt lgkmcnt(0)`），再发 `s_barrier`。

### 3.5 代码结构

```python
def kstep(acc, a_top, b_left, g, kt2, last):          # last: 0 稳态 / 1 不再 GR / 2 不再 GR、LW
    g_bl, g_at, g_ab, g_br = g
    acc[0] = dot(acc[0], a_top, b_left)                # region 0
    a_bot = lr(lds_a, 1, wm)                           #   LR A_bot(k)
    lw(lds_b, 0, g_bl)                                 #   LW B_left(k+1)
    g_bl = gr(B, 0, kt2)                               #   GR B_left(k+2)
    interleave(8, 4, 4)
    acc[1] = dot(acc[1], a_bot, b_left)                # region 1 ...
    ...
    gpu.barrier()                                      # X
    ...                                                # region 2, 3
    gpu.barrier()                                      # Y
    return acc, a_top, b_left, [g_bl, g_at, g_ab, g_br]

# prologue: tile0 -> LDS; tile1 -> 寄存器（顺序与循环一致）; LR b_left(0), a_top(0)
for k, st in range(fx.Index(0), fx.Index(NK - 2), fx.Index(1), init=state0):
    acc, a_top, b_left, g = unpack(st)                 # loop-carried：64 累加器 + 下一步操作数 + 16 暂存
    acc, a_top, b_left, g = kstep(acc, a_top, b_left, g, fx.Int32(k) + 2, 0)
    results = yield ...
kstep(..., last=1)                                     # peel：LW 最后一块，不再 GR
kstep(..., last=2)                                     # drain：只做 DOT
```

### 3.6 寄存器预算（每 lane，1 wave/SIMD）

| 文件 | 存放 | 个数 |
|---|---|---|
| AGPR 256 | C 累加器：4 象限 × 64×64 f32 / 64 lane | 256（满） |
| VGPR 256 | DOT 操作数：4 个半 tile × 每 wave 64×64 bf16 | 4 × 32 = 128 |
| | GR 暂存：4 个半 tile × 128×64 bf16 / 256 lane | 4 × 16 = 64 |
| | 地址、循环变量 | 其余 |

编译结果：`.vgpr_count 512`（含 AGPR 256），`.vgpr_spill_count 28`，**spill 只出现在循环外的 peel 段**；循环内没有 scratch。LLVM 会把一部分操作数放进 AGPR、一部分累加器放进 VGPR（gfx90a 以后 MFMA 可以直接读写 AGPR）。

这 64 个暂存 VGPR 是“经寄存器中转”的代价。NVIDIA Hopper 的 TMA 直接写 smem、wgmma 直接读 smem，所以没有这部分寄存器压力。

### 3.7 指令调度：`sched_group_barrier`

数据依赖只保证正确，不保证穿插：LLVM 可能把一个 region 的访存挤在一起。用 `rocdl.sched_group_barrier` 告诉调度器每个 region 的发射配比：

```python
def interleave(n_dsrd, n_dswr, n_vmem):
    for _ in range_constexpr(n_dsrd):
        rocdl.sched_mfma(2); rocdl.sched_dsrd(1)       # 每条 ds_read 配 2 条 MFMA
    for _ in range_constexpr(n_dswr):
        rocdl.sched_mfma(2); rocdl.sched_dswr(1)       # 每条 ds_write 配 2 条 MFMA
    for _ in range_constexpr(n_vmem):
        rocdl.sched_mfma(1); rocdl.sched_vmem(1)
    rocdl.sched_mfma(64 - 2 * n_dsrd - 2 * n_dswr - n_vmem)
```

- 为什么每次 LDS 访问配 2 条 MFMA：LDS 带宽 128 B/clk 由 4 个 SIMD 共享，一条 b128 访问（1 KB）平均每个 SIMD 占 32 cycle，正好是 2 条 16 cycle 的 MFMA。
- 编译后的循环体：256 MFMA、32 `ds_read_b128`、16 `ds_write_b128`、16 `global_load_dwordx4`、2 `s_barrier`。每次 ds_write 前是 `s_waitcnt vmcnt(15/14/13/12)`，也就是 16 条 global load 在途，每次只等最老的那几条。这正是 2 个 K 步的预取深度。
- prologue 的 GR 顺序必须和循环一致（B_left, A_top, A_bot, B_right）。否则 `SIInsertWaitcnts` 在循环入口合并两条路径的计数时会变保守，`vmcnt(15)` 会塌成 `vmcnt(3)` 甚至更小，流水被排空。

### 3.8 剩下的瓶颈

参考实现的消融实验：去掉循环里的 16 条 `ds_write_b128` 后，MFMA 效率从约 91% 升到 98%。一条 `ds_write_b128` 发射约需 20 cycle（4 地址 + 16 数据），而一条 MFMA 只有 16 cycle，所以一条 MFMA 盖不住一条 ds_write。这是 CDNA3“经寄存器中转”的固有成本，单个指令流里排不掉。

---

## 4. 结果

`bench.py`，单卡 MI300X（空闲时），HIP event 计时取 20 次中位数；计数器为 `rocprofv3 --pmc` 每次 dispatch 的值（`slides/measure.sh` 自动采集，原始数据在 `slides/data/`）：

| 版本 | 时间 (us) | TFLOPS | 相对上一版 | L2 命中 | L2 之外读取 | LDS bank 冲突 |
|---|---|---|---|---|---|---|
| v0 64×64 | 8051 | 162 | | | | |
| v0 128×128 | 3863 | 338 | 2.08x | 72.0% | 5.59 GB | 0 |
| v1 | 2965 | 440 | 1.30x | 74.7% | 1.24 GB | 0 |
| v2（不加 `sched_group_barrier`） | 2259 | 578 | 1.31x | | | |
| v2 | 2013 | **649** | 1.12x | 72.6% | 1.31 GB | 0 |
| torch.matmul（hipBLASLt，bf16 输出） | 2061 | 633 | | | | |

- **v0 64→128**：寄存器层强度 32 → 64 flop/B，性能约翻倍，说明 v0 受访存限制。
- **v1**：L2 之外（MALL/HBM）的读取从 5.59 GB 降到 1.24 GB；A+B 本身只有 0.29 GB，也就是每字节平均被读 4.2 次（v0 是 19 次）。这就是 CTA 级共享的效果。剩下的瓶颈是串行等待。
- **v2**：流水把访存藏到 MFMA 后面，比 v1 快 47%。其中 `sched_group_barrier` 让访存均匀穿插，单这一项贡献 12%。最终略高于 hipBLASLt（+2.4%），约为 2.1 GHz 峰值的 50%。满载时频率会低于 2.1 GHz，所以按实际频率算的 MFMA 效率更高。
- 外存流量和 L2 命中率在 v1、v2 之间基本不变：v2 的收益完全来自延迟隐藏，而不是更少的数据。

## 5. 下一步

- **inter_wave**：8 wave/CTA，每 SIMD 2 个 wave 分两组乒乓，一组在 MFMA 上时另一组做访存，把 ds_write 从关键路径上拿掉（参考实现约 99.7% MFMA 效率）。
- **XCD 感知的 CTA 重映射 + GROUP_SIZE_M**：CTA 按 id 轮流分到 8 个 XCD，每个 XCD 有独立的 L2。重排 CTA id，让同一 XCD 上的 CTA 读相同的 A 行和 B 行。
- **epilogue 加宽**：现在每个 lane 4 个标量 store。改用转置的 MFMA 布局（`transposed=True`），每个 lane 就持有同一行的 4 个连续 N，可以用 dwordx4。
- **buffer_load**：利用硬件越界检查，支持任意 M/N/K。
- **频率**：750 W 功耗墙下 MFMA 密集时会降频，比较 TFLOPS 时要考虑实际频率。

---

## 附录 A：CDNA3 与 Hopper 完整对照

MI300X（CDNA3，2023）同代的对手是 Hopper（H100 2022 / H200 2024）。

**算力密度**

| | MI300X（CDNA3） | H100 / H200 SXM（Hopper） |
|---|---|---|
| 计算单元 × 频率 | 304 CU × 2.1 GHz | 132 SM × ~1.83 GHz |
| 每单元矩阵核 | 4 个 Matrix Core（每 SIMD 1 个） | 4 个 Tensor Core（每子分区 1 个） |
| FP16/BF16 dense，FLOP / 单元 / cycle | **2048** | **4096** |
| FP8 dense，FLOP / 单元 / cycle | 4096 | 8192 |
| 整卡 BF16 dense | 1307 TFLOPS | 989 TFLOPS |
| FP32 vector（VALU / CUDA core），FLOP / 单元 / cycle | 128（`v_pk_fma_f32` 打包时 256） | 256（128 个 FP32 core） |
| FP64 vector，FLOP / 单元 / cycle | 128 | 128 |
| 整卡 FP32 vector | 81.7 TFLOPS（打包 163.4） | ~67 TFLOPS |
| 整卡 FP64 vector | 81.7 TFLOPS | ~34 TFLOPS |
| 矩阵 / vector 算力比（BF16 矩阵 ÷ FP32 vector） | 16×（打包 8×） | 16× |
| HBM | 5.3 TB/s（MI325X ~6 TB/s） | 3.35 TB/s（H100）/ 4.8 TB/s（H200） |

- 单个 CU 的矩阵算力是 Hopper 单个 SM 的一半。MI300X 整卡更高，是因为它有 304 个 CU，单元数是 H100 的 2.3 倍。
- vector 算力同理：未打包时单个 CU 是 Hopper SM 的一半；两者矩阵 / vector 都是 16×。CDNA3 的 FP64 vector 是全速（与 FP32 未打包相同），整卡 FP64 远高于 H100。
- 对 GEMM 来说 VALU 只用于地址计算和 epilogue，不是瓶颈；但 attention 的 softmax、量化 GEMM 的反量化这类在寄存器里做逐元素运算的 kernel，VALU 做 1 FLOP 的时间矩阵单元能做 16 FLOP，VALU 很容易反过来卡住 MFMA。

**执行模型与存储层次**

| | CDNA3 | Hopper（H100/H200） | 本例中的体现 |
|---|---|---|---|
| 调度单位 | wave64 | warp32；`wgmma` 以 warpgroup（4 warp = 128 线程）为单位 | MFMA 操作数排布以 64 lane 为单位（§1.2） |
| 执行宽度（FP32 VALU） | 每 CU 4 × SIMD16，wave64 一条指令占 4 cycle | 4 个子分区 × 32 FP32 lane，1 cycle | |
| 指令发射 | 4 个 SIMD 轮流，每个 SIMD 每 4 cycle 轮到一次；一次最多发 5 条不同类型（VALU / SALU / VMEM / LDS / 分支）、分别来自不同 wave，同一 wave 一次只发 1 条 | 每个子分区调度器每 cycle 从就绪 warp 中选 1 个发 1 条 | 1 wave/SIMD 时，每 4 cycle 最多发 1 条指令（§3.0） |
| 标量路径 | 独立 SALU + SGPR（每 wave ≤ 102 个），由编译器显式使用 | Uniform Datapath + 63 个 uniform 寄存器（Turing 起），编译器自动把 warp 内一致的值放进去 | 循环变量、基址、buffer 描述符在 SGPR |
| 寄存器 | SGPR / VGPR / AGPR 分离；VGPR+AGPR 每 CU 512 KB，每 wave 每 lane ≤ 512（256 V + 256 A），编译期固定 | 统一寄存器堆每 SM 256 KB（64K × 32 bit），每线程 ≤ 255；`setmaxnreg` 可在运行时按 warpgroup 调整（24~256） | 128×128/wave 的 fp32 累加器正好占满 256 AGPR |
| 矩阵指令 | MFMA，wave 级，操作数在 VGPR/AGPR | `wgmma.mma_async`，warpgroup 级，异步；B 直接读 smem，A 读 smem 或寄存器 | CDNA3 每个 K 步都要先 `ds_read` 到寄存器 |
| 片上共享存储 | LDS 64 KB/CU | smem ≤ 228 KB/SM（与 L1 共 256 KB） | NVIDIA 常见 3~7 级流水；CDNA3 只能 1 级（§3.2） |
| 异步拷贝 | 无宽的 global→LDS 拷贝（只有 4 B/lane） | TMA（`cp.async.bulk.tensor`）：单线程发起整块，硬件生成地址 | v2 用 64 个 VGPR 暂存下一块，并在指令流里发 `ds_write`（§3.6） |
| 等待机制 | `s_waitcnt vmcnt/lgkmcnt`：按发射顺序计数，编译器插入 | 编译器在指令控制位里编码 stall 数和 6 个依赖屏障（SB0~5），可变延迟指令按屏障号等；另有 `wgmma.wait_group`、mbarrier 事务计数 | 读 ISA 时看 `vmcnt(N)` 判断流水深度 |
| CTA 内同步 | 只有 `s_barrier`（全 CTA 所有 wave） | mbarrier（smem 中，arrive/wait 分离） + `bar.sync` | v2 每个 K 步 2 个全 CTA barrier（§3.4） |
| bank conflict | 32 bank × 4 B | 32 bank × 4 B；TMA 写 smem 时可由硬件做 32/64/128 B swizzle | XOR swizzle 原理相同（CUTLASS `Swizzle<3,4,3>`），CDNA3 要手算 |
| L2 | 4 MB/XCD × 8，XCD 之间不共享；另有 256 MB MALL | 50 MB，两个分区，全芯片一致 | CTA → XCD 映射会影响 L2 命中（§5） |
| CTA 间协作 | 无 | Thread Block Cluster：DSMEM、TMA multicast | |

## 附录 B：NVIDIA 经验在 CDNA3 上的常见误区

| NVIDIA 上的习惯 | CDNA3 上的情况 | 见 |
|---|---|---|
| 提高 occupancy，靠切 warp 藏延迟 | 高性能 GEMM 常态是 1 wave/SIMD，延迟只能靠 ILP | §1.3、§3.0 |
| 用 `ldmatrix` 把 smem 装成 fragment | 没有 `ldmatrix`，按 MFMA 排布直接用 `ds_read_b128` | §1.2 |
| scoreboard / 依赖屏障等指定的那条指令 | `s_waitcnt` 只能等最老的 N 条，发射顺序决定流水深度 | §1.4、§3.7 |
| `cp.async` / TMA + 3~7 级 smem 流水 | 没有宽的异步拷贝，LDS 只有 64 KB；用寄存器当第二级 buffer | §2.0、§3.2 |
| L2 全芯片共享 | 每个 XCD 独立 L2，CTA 编号决定落在哪个 XCD | §2.0、§5 |
| warp specialization + mbarrier + `setmaxnreg` | 没有部分 barrier 和运行时寄存器调整，不划算 | §3.0 |
| ptxas 会把访存和 MMA 排开 | LLVM 不保证穿插，需要 `sched_group_barrier` 显式指定 | §3.7 |
| 编译器自动利用 uniform datapath | 标量值要落在 SGPR（必要时 `readfirstlane`），否则占 VGPR 和 VALU | §1.0 |
