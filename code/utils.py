# utils.py

import os
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import center_of_mass

# ==========================================
# 🟢 数据流与格式转换工具
# ==========================================

def convert_cuda(item):
    """
    通用字典 CUDA 迁移工具
    将 Dataset 输出的 Tensor 字典批量推入 GPU 显存。
    """
    for key in item.keys():
        if key not in ['name', 'dst_name']:
            item[key] = item[key].float().cuda(non_blocking=True)
    return item

def save_nifti(image, path):
    """
    (后备函数) 使用 SimpleITK 保存 NIfTI 格式
    注：主流程在 eval.py 中已使用 nibabel 替代，此处保留作兼容性备用。
    """
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)

# ==========================================
# 🟢 物理量化评估度量 (Physics Metrics)
# ==========================================

def compute_com_error(mask_a, mask_b, spacing=(1.0, 1.0, 1.0)):
    """
    计算质心偏移误差 (Center of Mass Error, CoM)
    物理意义：用于评估两个 3D 解剖结构的宏观绝对位置差异。
    返回值：空间欧式距离 (默认单位：毫米 mm)。
    """
    # 提取所有概率大于 0.5 的体素位置计算质心
    com_a = np.array(center_of_mass(mask_a > 0.5))
    com_b = np.array(center_of_mass(mask_b > 0.5))

    # 异常兜底机制：如果预测出现了全黑 (空掩码)，惩罚性返回 999.0
    if np.any(np.isnan(com_a)) or np.any(np.isnan(com_b)):
        return 999.0

        # 结合图像的物理层厚 (spacing) 计算真实的物理距离
    physical_diff = (com_a - com_b) * np.array(spacing)
    return np.linalg.norm(physical_diff)

# ==========================================
# 🟢 临床级可视化渲染引擎 (Visualization Engine)
# ==========================================

def save_visualization_3view(img_np, prior_mask, aligned_prior, gt_mask, pred_mask, save_path, case_name, epoch):
    """
    三平面正交投影可视化 (Tri-planar Orthogonal Projections)
    核心功能：截取 3D 预测结果在 Axial(横断), Coronal(冠状), Sagittal(矢状)
    三个正交维度的中心切片，叠加于真实 MRI 灰度图像上。

    轮廓线定义：
    - 蓝色虚线 (Cyan dashed): 昨天的原始轮廓 (Initial Prior)
    - 绿色点线 (Lime dotted): 纯数学刚性对齐后的轮廓 (Aligned Prior)
    - 红色实线 (Red solid): 当天的真实标签 (Ground Truth Target)
    - 黄色实线 (Yellow solid): 算法最终推演的边界 (Network Prediction)

    学术意义：这张图将直接用于论文的 Results 章节，直观证明隐式网络在“绿色点线”
    的基础上，进一步捕获了局部非刚性形变（从而无限逼近“红色实线”）。
    """
    # 1. 定位空间质心，确保切片截取在器官的最丰满处
    coords = np.argwhere(gt_mask > 0.5)
    if len(coords) > 0:
        cx, cy, cz = coords.mean(axis=0).astype(int)
    else:
        cx, cy, cz = [s // 2 for s in gt_mask.shape]

    # 2. 从 128^3 的体积中提取三个维度的 2D 切片
    slices = [
        (img_np[:, :, cz], prior_mask[:, :, cz], aligned_prior[:, :, cz], gt_mask[:, :, cz], pred_mask[:, :, cz], "Axial (XY)"),
        (img_np[:, cy, :], prior_mask[:, cy, :], aligned_prior[:, cy, :], gt_mask[:, cy, :], pred_mask[:, cy, :], "Coronal (XZ)"),
        (img_np[cx, :, :], prior_mask[cx, :, :], aligned_prior[cx, :, :], gt_mask[cx, :, :], pred_mask[cx, :, :], "Sagittal (YZ)")
    ]

    # 3. 构建 1x3 的并排画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Case: {case_name} | Epoch: {epoch}", fontsize=16)

    for i, (img_slice, prior_slc, aligned_slc, gt_slc, pred_slc, title) in enumerate(slices):
        ax = axes[i]
        # 显示高对比度的 MRI 底图
        ax.imshow(img_slice, cmap='gray', origin='lower')

        # 防御性绘制：必须确认当前切片内存在前景轮廓，否则 contour 函数会报错
        if prior_slc.sum() > 0:
            ax.contour(prior_slc, levels=[0.5], colors='cyan', linestyles='dashed', linewidths=1.5, alpha=0.7)
        if aligned_slc.sum() > 0:
            ax.contour(aligned_slc, levels=[0.5], colors='lime', linestyles='dotted', linewidths=2.0, alpha=0.9)
        if gt_slc.sum() > 0:
            ax.contour(gt_slc, levels=[0.5], colors='red', linestyles='solid', linewidths=2.0)
        if pred_slc.sum() > 0:
            ax.contour(pred_slc, levels=[0.5], colors='yellow', linestyles='solid', linewidths=2.0)

        ax.set_title(title)
        ax.axis('off') # 隐藏坐标轴刻度

    # 4. 添加规范的学术图例 (Legend)
    custom_lines = [
        Line2D([0], [0], color='cyan', lw=2, linestyle='dashed'),
        Line2D([0], [0], color='lime', lw=2, linestyle='dotted'),
        Line2D([0], [0], color='red', lw=2),
        Line2D([0], [0], color='yellow', lw=2)
    ]
    fig.legend(custom_lines,
               ['Original Prior (Init Pos)', 'Aligned Prior (Rigid Baseline)', 'Ground Truth (Target)', 'Prediction (Our Model)'],
               loc='lower center', ncol=4, fontsize=12)

    # 挤压边缘空白，为图例留出底部空间
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)