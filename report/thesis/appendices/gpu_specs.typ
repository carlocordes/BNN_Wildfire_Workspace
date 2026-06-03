#import "../template.typ": *

= Computational Specifications

#align(center)[
  #table(
    columns: (1.5fr, 3fr),
    inset: 9pt,
    align: (left, left),
    stroke: 0.5pt + luma(120),
    fill: (x, y) => if y == 0 { luma(240) } else { none },
    
    // Table Header
    [*Hardware Parameter*], [*System Specification*],
    
    [GPU Compute Controller], [NVIDIA Corporation GA102 [GeForce RTX 3090] (rev a1)],
    [Subsystem Vendor], [ASUSTeK Computer Inc. (Device 87b3)],
    [BAR1 Memory (VRAM Window)], [32 GB (64-bit, prefetchable, address: #raw("b000000000"))],
    [MMIO Memory Space], [16 MB (32-bit, non-prefetchable, address: #raw("c2000000"))],
    [BAR2 Memory Space], [32 MB (64-bit, prefetchable, address: #raw("b800000000"))],
    [I/O Ports], [Size: 128 (Base address: #raw("6000"))],
    [Active Kernel Driver], [#raw("nvidia")],
    [Available Kernel Modules], [#raw("nouveau"), #raw("nvidia_drm"), #raw("nvidia")],
    [PCIe Properties], [Bus Master, Fast Devsel, Latency: 0, IRQ: 266],
    [IOMMU Allocation], [Group 45]
  )
]

