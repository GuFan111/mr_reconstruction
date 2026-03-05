# eval.py

import os
# 限制底层数学库的线程数，防止多进程评估时的 CPU 资源抢占
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import nibabel as nib  # 🟢 引入 NIfTI 格式保存库，用于与标准临床医学影像软件 (如 ITK-SNAP) 对接

# 引入 MONAI 的评估度量工具
from monai.metrics import compute_hausdorff_distance, compute_dice, compute_average_surface_distance
from scipy.ndimage import center_of_mass, shift

from dataset import Prostate_Dataset
from models.model import DIF_Net
from utils import compute_com_error, save_visualization_3view

# ==========================================
#  🟢 全量程评估配置区域 (Full-scale Evaluation Config)
#  设计目标：在不进行任何数据增强的情况下，测试模型在连续物理空间的泛化拓扑重构能力
# ==========================================
class EvalConfig:
    name = 'prostate_eval_new_1'
    data_root = r'/root/autodl-tmp/Proj/data/prostate_158_128'
    checkpoint_path = r'/root/autodl-tmp/Proj/code/logs/prostate_new_1/model_best.pth'
    gpu_id = 0
    num_views = 3
    out_res = (128, 128, 128)
    combine = 'attention'
    save_vis = True  # 是否导出 2D 轮廓叠加图 (用于论文 Main Figure)
    save_nii = True  # 🟢 是否导出 NIfTI 格式 (用于临床 3D 渲染与表面误差热力图分析)

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='w') # 注意：Eval 是覆盖写入 'w'
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# 🟢 便捷的 NIfTI 临床格式保存器
def save_nifti(array_np, save_path, is_mask=True):
    """
    将 Numpy 矩阵打包为标准的 NIfTI 医学图像格式。
    学术意义：这使得算法预测的靶区可以无缝导回医院的放疗计划系统 (TPS)。
    """
    # 标签转为 uint8 节省空间，灰度图保持 float32
    dtype = np.uint8 if is_mask else np.float32
    # 默认采用单位仿射矩阵 (np.eye)，保持相对物理体素空间
    nii_img = nib.Nifti1Image(array_np.astype(dtype), np.eye(4))
    nib.save(nii_img, save_path)

if __name__ == '__main__':
    # ==========================================
    # 初始化与环境隔离
    # ==========================================
    os.environ['CUDA_VISIBLE_DEVICES'] = str(EvalConfig.gpu_id)
    save_dir = f'./logs/{EvalConfig.name}'
    os.makedirs(save_dir, exist_ok=True)
    if EvalConfig.save_vis:
        os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)
    if EvalConfig.save_nii:
        os.makedirs(os.path.join(save_dir, 'vis_nii'), exist_ok=True)

    logger = setup_logger(os.path.join(save_dir, 'eval_log.txt'))
    logger.info(f"Start Evaluation on entire Validation Set...")
    logger.info(f"Loading checkpoint from: {EvalConfig.checkpoint_path}")

    # ==========================================
    # 数据与模型加载 (Eval 模式无乱序，Batch=1)
    # ==========================================
    val_dst = Prostate_Dataset(data_root=os.path.join(EvalConfig.data_root, 'valid', 'image'),
                               label_root=os.path.join(EvalConfig.data_root, 'valid', 'label'),
                               split='eval', npoint=50000, out_res=EvalConfig.out_res)
    eval_loader = DataLoader(val_dst, batch_size=1, shuffle=False)

    model = DIF_Net(num_views=EvalConfig.num_views, combine=EvalConfig.combine).cuda()

    if not os.path.exists(EvalConfig.checkpoint_path):
        logger.error(f"Checkpoint not found: {EvalConfig.checkpoint_path}")
        exit()

    checkpoint = torch.load(EvalConfig.checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint)
    model.eval()

    logger.info(f"Model loaded successfully. Total validation samples: {len(val_dst)}")

    # ==========================================
    # 🟢 临床量化指标矩阵 (Clinical Metric Matrix)
    # ==========================================
    metrics = {
        'com_init': [], 'com_pred': [],          # CoM: 评估宏观空间漂移 (Spatial Drift)
        'dice_init': [], 'dice_align': [], 'dice_pred': [], # Dice: 评估体积级交并比 (Volumetric Overlap)
        'hd95_init': [], 'hd95_pred': [],        # HD95: 放疗最敏感指标，评估最差 5% 的边界异常膨胀/塌陷
        'assd_init': [], 'assd_pred': [],        # ASSD: 评估全局表面的平均亚体素级贴合度
        'vol_diff_ratio': [],                    # Vol_Diff: 评估生物力学拓扑守恒定律 (防止过度萎缩/膨胀)
        'inference_time': []                     # Time: 评估在线 MR-Linac 场景的实时性要求 (<0.5秒)
    }

    with torch.no_grad():
        with tqdm(eval_loader, desc='Evaluating', ncols=100) as pbar:
            for i, v_item in enumerate(pbar):
                for key in v_item.keys():
                    if key not in ['name']: v_item[key] = v_item[key].float().cuda(non_blocking=True)

                torch.cuda.synchronize()
                t_start = time.time()

                # ==========================================
                # 🟢 连续空间分块推理 (Chunked Inference)
                # 由于网络直接在坐标系中预测概率，这里调度了 eval_npoint 防止显存爆显
                # ==========================================
                pred_logits = model(v_item, is_eval=True, eval_npoint=50000)

                torch.cuda.synchronize()
                t_end = time.time()
                metrics['inference_time'].append(t_end - t_start)

                # ==========================================
                # 隐式点云向 3D 物理矩阵流形的拓扑还原
                # ==========================================
                res = EvalConfig.out_res
                pred_probs = torch.sigmoid(pred_logits)
                pred_mask_tensor = (pred_probs > 0.5).float().view(1, 1, res[0], res[1], res[2])
                gt_mask_tensor = v_item['p_gt'].view(1, 1, res[0], res[1], res[2])
                prior_mask_tensor = v_item['prior_mask'] # 注意：此处是已被 Dataset 预对齐后的 Prior

                pred_mask_np = pred_mask_tensor[0, 0].cpu().numpy()
                gt_mask_np = gt_mask_tensor[0, 0].cpu().numpy()
                prior_mask_np = prior_mask_tensor[0, 0].cpu().numpy()

                # ------------------------------------------
                # 计算刚性对齐理论基线 (Rigid Alignment Baseline)
                # 注：因 Dataset 已做平移，此处的 shift_vec 通常极小，Init Dice ≈ Aligned Dice
                # ------------------------------------------
                try:
                    com_gt = np.array(center_of_mass(gt_mask_np))
                    com_prior = np.array(center_of_mass(prior_mask_np))
                    shift_vec = com_gt - com_prior
                    aligned_prior_np = shift(prior_mask_np, shift_vec, order=0)
                    aligned_prior_tensor = torch.from_numpy(aligned_prior_np).unsqueeze(0).unsqueeze(0).cuda()
                except:
                    aligned_prior_tensor = prior_mask_tensor
                    aligned_prior_np = prior_mask_np

                # ------------------------------------------
                # 执行核心度量计算
                # ------------------------------------------
                init_com_err = compute_com_error(gt_mask_np, prior_mask_np, spacing=(1.0, 1.0, 1.0))
                recon_com_err = compute_com_error(gt_mask_np, pred_mask_np, spacing=(1.0, 1.0, 1.0))
                metrics['com_init'].append(init_com_err); metrics['com_pred'].append(recon_com_err)

                gt_b, pred_b, prior_b = gt_mask_tensor, pred_mask_tensor, prior_mask_tensor
                i_dice = compute_dice(prior_b, gt_b, include_background=False).item()
                a_dice = compute_dice(aligned_prior_tensor, gt_b, include_background=False).item()
                r_dice = compute_dice(pred_b, gt_b, include_background=False).item()
                metrics['dice_init'].append(i_dice); metrics['dice_align'].append(a_dice); metrics['dice_pred'].append(r_dice)

                # 表面距离指标 (带有异常兜底机制)
                try: i_hd95 = compute_hausdorff_distance(prior_b, gt_b, include_background=False, percentile=95).item()
                except: i_hd95 = 99.0
                try: r_hd95 = compute_hausdorff_distance(pred_b, gt_b, include_background=False, percentile=95).item()
                except: r_hd95 = 99.0
                metrics['hd95_init'].append(i_hd95); metrics['hd95_pred'].append(r_hd95)

                try: i_assd = compute_average_surface_distance(prior_b, gt_b, include_background=False, symmetric=True).item()
                except: i_assd = 99.0
                try: r_assd = compute_average_surface_distance(pred_b, gt_b, include_background=False, symmetric=True).item()
                except: r_assd = 99.0
                metrics['assd_init'].append(i_assd); metrics['assd_pred'].append(r_assd)

                v_gt = gt_mask_np.sum()
                v_pred = pred_mask_np.sum()
                vol_diff_ratio = abs(v_pred - v_gt) / (v_gt + 1e-5) * 100.0
                metrics['vol_diff_ratio'].append(vol_diff_ratio)

                logger.info(f"[{v_item['name'][0]}] Dice(Init/Align/Pred): {i_dice:.3f}/{a_dice:.3f}/{r_dice:.3f} | HD95: {r_hd95:.2f}mm | ASSD: {r_assd:.2f}mm | Vol_Diff: {vol_diff_ratio:.1f}%")

                # ==========================================
                # 🟢 临床下游可视化导出引擎
                # ==========================================
                if EvalConfig.save_vis:
                    vis_save_path = os.path.join(save_dir, 'vis', f"{v_item['name'][0]}_pred_{r_dice:.3f}.png")
                    gt_img_np = v_item['target_image'][0, 0].cpu().numpy()
                    save_visualization_3view(
                        img_np=gt_img_np,
                        prior_mask=prior_mask_np,
                        aligned_prior=aligned_prior_np,
                        gt_mask=gt_mask_np,
                        pred_mask=pred_mask_np,
                        save_path=vis_save_path,
                        case_name=v_item['name'][0],
                        epoch="Best"
                    )

                if EvalConfig.save_nii:
                    case_id = v_item['name'][0]
                    nii_dir = os.path.join(save_dir, 'vis_nii')

                    # 1. 导出 MRI 灰度原图作为底图 (提供绝对解剖学参考系)
                    gt_img_np = v_item['target_image'][0, 0].cpu().numpy()
                    save_nifti(gt_img_np, os.path.join(nii_dir, f"{case_id}_IMG.nii.gz"), is_mask=False)

                    # 2. 导出各类 Mask，以便在 ITK-SNAP 中生成多通道 Overlay 与 3D 渲染图
                    save_nifti(gt_mask_np, os.path.join(nii_dir, f"{case_id}_GT.nii.gz"))
                    save_nifti(prior_mask_np, os.path.join(nii_dir, f"{case_id}_PRIOR.nii.gz"))
                    save_nifti(pred_mask_np, os.path.join(nii_dir, f"{case_id}_PRED_{r_dice:.3f}.nii.gz"))

    # ==========================================
    # 全量程终态数据统计汇总
    # ==========================================
    logger.info("\n" + "="*50)
    logger.info("FINAL EVALUATION RESULTS (Mean ± Std)")
    logger.info("="*50)
    logger.info(f" - CoM Error (mm)      : Init {np.mean(metrics['com_init']):.2f} ± {np.std(metrics['com_init']):.2f} -> Pred {np.mean(metrics['com_pred']):.2f} ± {np.std(metrics['com_pred']):.2f}")
    logger.info(f" - Dice Score          : Init {np.mean(metrics['dice_init']):.4f} | Aligned {np.mean(metrics['dice_align']):.4f} | Pred {np.mean(metrics['dice_pred']):.4f} ± {np.std(metrics['dice_pred']):.4f}")
    logger.info(f" - HD95 (mm)           : Init {np.mean(metrics['hd95_init']):.2f} -> Pred {np.mean(metrics['hd95_pred']):.2f} ± {np.std(metrics['hd95_pred']):.2f}")
    logger.info(f" - ASSD (mm)           : Init {np.mean(metrics['assd_init']):.2f} -> Pred {np.mean(metrics['assd_pred']):.2f} ± {np.std(metrics['assd_pred']):.2f}")
    logger.info(f" - Volume Diff Ratio   : {np.mean(metrics['vol_diff_ratio']):.2f}% ± {np.std(metrics['vol_diff_ratio']):.2f}%")
    logger.info(f" - Inference Time      : {np.mean(metrics['inference_time']):.3f} s/vol")
    logger.info("="*50)