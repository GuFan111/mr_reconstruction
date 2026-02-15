import torch
import torch.nn.functional as F
import time
import numpy as np
import scipy.ndimage as ndimage
from monai.inferers import sliding_window_inference
from models.mednext.create_mednext_v1 import create_mednext_v1

# ================= 配置区域 =================
class Config:
    # 模拟输入数据的形状 [Batch, Channel, D, H, W] (AMOS MRI 典型尺寸)
    input_shape = (1, 1, 100, 256, 256) 
    
    num_classes = 16
    
    # 训练时的 Patch Size (注意顺序: X, Y, Z)
    # 确保 fast_inference 的 input_size 与此一致
    crop_size = (160, 160, 96) 
    
    overlap = 0.1               # 滑动窗口重叠率
    sw_batch_size = 4           # 滑动窗口 Batch Size
    model_id = 'S'              # Model Size

# ================= 🐌 旧方案: CPU 后处理 (慢) =================
def cpu_post_process(pred_tensor):
    """
    传统的 CPU 后处理管道
    """
    # 1. GPU -> CPU (耗时!)
    pred_np = torch.argmax(pred_tensor, dim=1).cpu().numpy()[0]
    
    # 2. Scipy 连通域/闭运算 (CPU计算, 慢!)
    # 模拟一个简单的闭运算
    struct = ndimage.generate_binary_structure(3, 1)
    pred_np = ndimage.binary_closing(pred_np, structure=struct, iterations=1)
    
    return pred_np

# ================= 🚀 新方案: 纯 GPU 管道 (快) =================
class FastROIPipeline:
    def __init__(self, model, input_size=(160, 160, 96)):
        self.model = model
        self.input_size = input_size
        
    def gpu_morphology_closing(self, mask_tensor, kernel_size=5):
        """
        在 GPU 上执行形态学闭运算 (填补断裂)
        输入: [B, 1, D, H, W] 的 0/1 Float Tensor
        """
        # Padding 计算: kernel_size // 2
        pad = kernel_size // 2
        
        # 1. 膨胀 (Dilation) - 连接断裂
        # MaxPool3d 相当于膨胀
        dilated = F.max_pool3d(mask_tensor, kernel_size=kernel_size, stride=1, padding=pad)
        
        # 2. 腐蚀 (Erosion) - 恢复原大小 
        # -MaxPool3d(-x) 相当于腐蚀
        closed = -F.max_pool3d(-dilated, kernel_size=kernel_size, stride=1, padding=pad)
        
        return closed

    def run(self, inputs):
        """
        全流程 GPU 推理 (不回传 CPU)
        """
        original_size = inputs.shape[2:]
        
        # 1. 降采样 (GPU)
        inputs_small = F.interpolate(inputs, size=self.input_size, mode='area')
        
        # 2. 模型推理 (GPU + FP16)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                logits_small = self.model(inputs_small)
        
        # 3. 上采样 (GPU)
        logits_large = F.interpolate(logits_small, size=original_size, mode='trilinear', align_corners=False)
        
        # 4. 生成 Mask (GPU)
        pred_mask = torch.argmax(logits_large, dim=1, keepdim=True).float()
        
        # 5. 形态学后处理 (GPU)
        final_mask = self.gpu_morphology_closing(pred_mask, kernel_size=5)
        
        return final_mask

# ================= 主函数 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running speed test on device: {device}")
    print(f"Input Shape: {Config.input_shape}")
    
    # 1. 初始化模型
    model = create_mednext_v1(
        num_input_channels=1,
        num_classes=Config.num_classes,
        model_id=Config.model_id,
        kernel_size=3,
        deep_supervision=False
    ).to(device)
    model.eval()

    # 2. 准备数据
    dummy_input = torch.randn(Config.input_shape).to(device)
    
    # 3. 初始化快速管道
    fast_pipeline = FastROIPipeline(model, input_size=Config.crop_size)

    # ---------------------------------------------------------
    # 阶段 1: 预热 (Warm-up)
    # ---------------------------------------------------------
    print("\n[Phase 1] Warming up GPU...")
    with torch.no_grad():
        for _ in range(10):
            _ = fast_pipeline.run(dummy_input)
    torch.cuda.synchronize()

    # ---------------------------------------------------------
    # 阶段 2: 测试基准 (Sliding Window + CPU Post-Process)
    # ---------------------------------------------------------
    print(f"\n🐌 [Baseline] Sliding Window + CPU Post-Process:")
    print(f"   (This simulates your OLD method)")
    
    start_time = time.time()
    loops_slow = 3 # 跑慢点，跑多了浪费时间
    
    with torch.no_grad():
        for _ in range(loops_slow):
            # A. 推理
            output = sliding_window_inference(
                dummy_input, roi_size=Config.crop_size, 
                sw_batch_size=Config.sw_batch_size, predictor=model, overlap=Config.overlap
            )
            # B. CPU 后处理
            _ = cpu_post_process(output)
            
    torch.cuda.synchronize() # 确保 CPU 任务也完成了
    end_time = time.time()
    
    avg_slow = (end_time - start_time) / loops_slow
    print(f"   ⏱️  Average Time: {avg_slow:.4f} s ({avg_slow*1000:.1f} ms)")

    # ---------------------------------------------------------
    # 阶段 3: 测试优化方案 (Global Resize + GPU Post-Process)
    # ---------------------------------------------------------
    print(f"\n🚀 [Optimized] Global Resize + GPU Post-Process:")
    print(f"   (This simulates your NEW method for MR-Linac)")
    
    start_time = time.time()
    loops_fast = 100 # 速度快，多跑点求平均
    
    with torch.no_grad():
        for _ in range(loops_fast):
            # 全流程都在 GPU 上
            _ = fast_pipeline.run(dummy_input)
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_fast = (end_time - start_time) / loops_fast
    print(f"   ⏱️  Average Time: {avg_fast:.4f} s ({avg_fast*1000:.1f} ms)")
    print(f"   🔥 FPS: {1/avg_fast:.1f}")

    # ---------------------------------------------------------
    # 总结
    # ---------------------------------------------------------
    print("-" * 40)
    print(f"📊 Speedup Factor: {avg_slow / avg_fast:.1f}x Faster")
    if avg_fast < 0.05:
        print("✅ Status: Real-time Requirement (<50ms) MET!")
    else:
        print("⚠️ Status: Still optimization needed.")

if __name__ == '__main__':
    main()