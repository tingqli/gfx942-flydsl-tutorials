# FlyDSL functions

FlyDSL的对象，函数都有什么作用，其行为在最终汇编会如何表现，我们尝试针对目标FlyDSL的特性进行实验，力求搞清楚其含义，用法，最终汇编行为。

每个实验一个目录或者一个.ipynb文件，目录名/notebook名就是实验名。

如果是目录的话其中包含：

 - README.md：先写结论，再写实验细节；
 - 实验脚本；
 - results/：原始输出和数据。

如果是 .ipynb文件就循序渐进深入浅出的使用非常简单直观的例子讲解实验内容和结论。

| 实验 | 概要 |
|---|---|
| [tensor_101](tensor_101.ipynb) | `fx.Tensor` = 指针 + layout；torch 传入的动态 layout 与 `make_view` 换成的静态 layout 在 ISA 上的差别；`.load()/.store()` 的指令宽度由子 tensor 是否连续决定。 |
| [layout_algebra](layout_algebra.ipynb) | 用 FlyDSL 演示 CuTe layout 代数（coalesce / complement / composition / divide / product / right_inverse）；运算只改类型、不生成指令，但决定访存宽度；v0 的 A layout 就是一次 `logical_divide`。 |
| [LDS_101](lds_101.ipynb) | 用 `@fx.struct` 定义 LDS 结构并分配；LDS tensor 的 `ds_*` 指令；bank conflict 与 `Swizzle` 的几种写法；HBM→LDS DMA 的各种接口、同步/异步等待，以及封装 `dma_to_lds`。ISA 对比 gfx942 / gfx950。 |
| [register_101](register_101.ipynb) | `.load()` 之后的寄存器类型（`fx.Vector`、标量 `Numeric`、寄存器 tensor）及其自带操作的 ISA；跨 lane 操作（shuffle、DPP、ballot）；`fx.rocdl` 专用指令；用 `llvm.call_intrinsic`、`vector.insert`、`llvm.inline_asm` 补上没有封装的功能。ISA 对比 gfx942 / gfx950。 |
| [kernel_101](kernel_101.ipynb) | kernel 文件的组织方式：`@flyc.kernel` 与 `@flyc.jit` 的分工；trace 与 JIT 缓存 key；为什么用 `lru_cache` 修饰的 `build(...)` 传编译期参数；AST 改写对 `if`/`for` 的影响与常见坑；几种 for loop 写法生成的 IR/ISA；子函数加不加 `@flyc.jit` 的区别；kernel 里为什么不能 launch kernel。 |