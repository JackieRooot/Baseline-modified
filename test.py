"""
GPU 训练瓶颈诊断脚本
运行后根据输出判断瓶颈所在
"""
import time, os, sys

def check_env():
    try:
        import torch
    except ImportError:
        print("❌ 未安装 PyTorch"); sys.exit(1)

    print("=" * 55)
    print("  GPU 训练瓶颈诊断报告")
    print("=" * 55)

    cuda = torch.cuda.is_available()
    print(f"CUDA 可用:        {'是' if cuda else '否'}")
    if not cuda:
        print("❌ 没有可用 GPU，无法诊断"); return

    prop = torch.cuda.get_device_properties(0)
    print(f"GPU:              {prop.name}")
    print(f"显存:             {prop.total_memory/1024**3:.1f} GB")
    print(f"SM 数量:          {prop.multi_processor_count}")
    print()

    # ── 1. 纯 GPU 算力基准 ─────────────────────────────
    print("【1】纯 GPU 算力（排除 CPU/IO 瓶颈）")
    size = 4096
    a = torch.randn(size, size, device='cuda', dtype=torch.float16)
    b = torch.randn(size, size, device='cuda', dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tflops = 50 * 2 * size**3 / elapsed / 1e12
    print(f"  矩阵乘法 TFLOPS: {tflops:.2f}  (4060 理论峰值 ~16 TFLOPS fp16)")
    print(f"  GPU 峰值利用率估算: {min(100, tflops/16*100):.0f}%")
    print()

    # ── 2. CPU→GPU 传输速度 ───────────────────────────
    print("【2】CPU → GPU 数据传输速度")
    data = torch.randn(256, 3, 224, 224)  # 模拟一批图像
    t0 = time.perf_counter()
    for _ in range(20):
        _ = data.to('cuda', non_blocking=False)
        torch.cuda.synchronize()
    sync_ms = (time.perf_counter() - t0) / 20 * 1000

    t0 = time.perf_counter()
    for _ in range(20):
        _ = data.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
    async_ms = (time.perf_counter() - t0) / 20 * 1000

    mb = data.nbytes / 1024**2
    print(f"  单批大小:         {mb:.1f} MB  (batch=256, 224x224 图像)")
    print(f"  同步传输耗时:     {sync_ms:.1f} ms")
    print(f"  异步传输耗时:     {async_ms:.1f} ms")
    bw = mb / (async_ms / 1000) / 1024
    print(f"  估算传输带宽:     {bw:.1f} GB/s")
    if bw < 5:
        print("  ⚠️  传输带宽偏低，可能是 PCIe 瓶颈或未用 pin_memory")
    print()

    # ── 3. DataLoader 吞吐估算 ───────────────────────
    print("【3】DataLoader 配置建议（基于当前硬件）")
    import multiprocessing
    cpu_cores = multiprocessing.cpu_count()
    print(f"  CPU 核心数:       {cpu_cores}")
    recommended_workers = min(cpu_cores, 8)
    print(f"  建议 num_workers: {recommended_workers}")
    print(f"  建议 pin_memory:  True  （CUDA 训练时必开）")
    print(f"  建议 prefetch_factor: 2~4")
    print()

    # ── 4. 显存使用情况 ───────────────────────────────
    print("【4】当前显存使用")
    alloc = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    total = prop.total_memory / 1024**2
    print(f"  已分配:   {alloc:.0f} MB")
    print(f"  已预留:   {reserved:.0f} MB")
    print(f"  总显存:   {total:.0f} MB")
    util = reserved / total * 100
    if util < 50:
        print(f"  ⚠️  显存占用仅 {util:.0f}%，batch size 可能太小，GPU 喂不饱")
    print()

    # ── 5. 综合诊断建议 ──────────────────────────────
    print("=" * 55)
    print("  常见原因 & 修复方向")
    print("=" * 55)
    tips = [
        ("DataLoader num_workers=0", "改为 num_workers=4~8，开启多进程预加载"),
        ("未开 pin_memory",          "DataLoader 加 pin_memory=True"),
        ("batch size 太小",           "增大 batch size，让 GPU 每次处理更多数据"),
        ("CPU 预处理太慢",            "减少 transform 复杂度，或移到 GPU 做"),
        ("模型太小 / 层数少",         "小模型计算量不足以喂饱 GPU"),
        ("混合精度未开启",            "用 torch.cuda.amp.autocast() 加速"),
        ("调试模式 / 频繁 .item()",   "训练循环内避免 .item() / print 同步操作"),
    ]
    for cause, fix in tips:
        print(f"  原因: {cause}")
        print(f"  修复: {fix}")
        print()

check_env()