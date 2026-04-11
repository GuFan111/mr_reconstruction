import os
# 限制底层数学库的线程数，切断多进程评估时的 CPU 资源灾难性抢占
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import nibabel as nib

from monai.metrics import compute_hausdorff_distance
from scipy.ndimage import center_of_mass, shift

from dataset import Prostate_Dataset
from models.model import DIF_Net
from models.baseline_models import Baseline_3DUNet, Baseline_SwinUNETR
from utils import save_visualization_3view

# ==========================================
#  🟢 法医级绝对信息隔离算子
# ==========================================
def compute_isolated_dice(pred_np, gt_np, mask_purged):
    """在绝对信息剥夺的 97% 盲区内计算拓扑连通性"""
    p = pred_np[mask_purged].astype(bool)
    g = gt_np[mask_purged].astype(bool)
    union = p.sum() + g.sum()
    if union == 0: return np.nan
    return 2.0 * np.logical_and(p, g).sum() / union

def safe_global_hd95(pred_tensor, gt_tensor):
    """计算全局几何边界误差，免疫局部切割导致的拓扑撕裂伪影"""
    if pred_tensor.sum() == 0 or gt_tensor.sum() == 0: return 99.0
    try:
        return compute_hausdorff_distance(pred_tensor, gt_tensor, include_background=False, percentile=95).item()
    except:
        return 99.0

# ==========================================
#  🟢 全量程测试配置区域 (Test-Set Config)
# ==========================================
class EvalConfig:
    # 🔴 算法处决开关: 'difnet', '3dunet', 'swin_unetr'
    # 注意：Oracle 不再是独立模型，它将作为刚性物理基底与所选模型并发输出
    model_type = 'difnet'

    name = f'prostate_4_8_extreme_local_attention_{model_type}'
    data_root = r'/root/autodl-tmp/Proj/data/prostate_158_128'

    # 🔴 强制权重路由映射 (Hard-coded Weights Routing)
    model_weights = {
        'difnet': r'/root/autodl-tmp/Proj/code/logs/prostate_4_8_attention/model_best.pth',
        # 'difnet': r'/root/autodl-tmp/Proj/code/logs/prostate_4_8_mlp/model_best.pth',
        '3dunet': r'/root/autodl-tmp/Proj/code/logs/baseline_3dunet_sparse_amp/model_best.pth',
        'swin_unetr': r'/root/autodl-tmp/Proj/code/logs/baseline_swin_unetr_sparse_amp/model_best.pth'
    }

    gpu_id = 0
    num_views = 3
    out_res = (128, 128, 128)
    combine = 'attention'
    save_vis = True

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(EvalConfig.gpu_id)
    save_dir = f'./logs/{EvalConfig.name}'
    os.makedirs(save_dir, exist_ok=True)
    if EvalConfig.save_vis:
        os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)

    logger = setup_logger(os.path.join(save_dir, 'test_log.txt'))
    logger.info(f"[FORENSIC INITIATION] Absolute Information Isolation Protocol Activated.")
    logger.info(f"Target Architecture: {EvalConfig.model_type.upper()}")

    # 获取对应权重路径并校验合法性
    checkpoint_path = EvalConfig.model_weights.get(EvalConfig.model_type)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.error(f"致命错误：未找到模型 {EvalConfig.model_type} 的有效权重路径 -> {checkpoint_path}")
        exit()
    logger.info(f"Loading weights from: {checkpoint_path}")

    test_dst = Prostate_Dataset(data_root=os.path.join(EvalConfig.data_root, 'test', 'image'),
                                label_root=os.path.join(EvalConfig.data_root, 'test', 'label'),
                                split='test', npoint=50000, out_res=EvalConfig.out_res)
    test_loader = DataLoader(test_dst, batch_size=1, shuffle=False)

    # 模型架构实例化
    if EvalConfig.model_type == 'difnet':
        model = DIF_Net(num_views=EvalConfig.num_views, combine=EvalConfig.combine).cuda()
    elif EvalConfig.model_type == '3dunet':
        model = Baseline_3DUNet().cuda()
    elif EvalConfig.model_type == 'swin_unetr':
        model = Baseline_SwinUNETR().cuda()

    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    if 'net.weight' in list(checkpoint.keys())[0] and EvalConfig.model_type != 'difnet':
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 建立并发核算矩阵 (Concurrent Metric Matrix)
    metrics = {
        'oracle_iso_dice': [], 'pred_iso_dice': [],
        'oracle_glo_hd95': [], 'pred_glo_hd95': [],
        'inference_time': []
    }

    with torch.no_grad():
        with tqdm(test_loader, desc='Forensic Execution', ncols=100) as pbar:
            for i, v_item in enumerate(pbar):
                # 张量挂载
                for key in v_item.keys():
                    if key not in ['name']: v_item[key] = v_item[key].float().cuda(non_blocking=True)

                torch.cuda.synchronize()
                t_start = time.time()

                # ==========================================
                # 🔴 轨迹一：非刚性算法流形推演
                # ==========================================
                if EvalConfig.model_type == 'difnet':
                    pred_logits = model(v_item, is_eval=True, eval_npoint=50000)
                else:
                    full_target_img = v_item['target_image'].float()
                    full_prior_img = v_item['prior_image'].float()
                    prior_mask_tensor = v_item['prior_mask'].float()
                    coords = v_item['center_coords']
                    with torch.amp.autocast('cuda'):
                        pred_logits = model(full_target_img, full_prior_img, prior_mask_tensor, coords)

                torch.cuda.synchronize()
                metrics['inference_time'].append(time.time() - t_start)

                pred_mask_np = (torch.sigmoid(pred_logits) > 0.5).float().view(128, 128, 128).cpu().numpy()

                # ==========================================
                # 🔴 轨迹二：Oracle 并发刚性物理对齐
                # ==========================================
                gt_mask_np = v_item['p_gt'].view(128, 128, 128).cpu().numpy()
                prior_mask_np = v_item['prior_mask'].view(128, 128, 128).cpu().numpy()

                com_gt = np.array(center_of_mass(gt_mask_np))
                com_prior = np.array(center_of_mass(prior_mask_np))
                shift_vec = com_gt - com_prior
                oracle_aligned_np = shift(prior_mask_np, shift_vec, order=0)

                # ==========================================
                # 🔴 构建狄利克雷物理边界隔离掩码 (~M_anchor)
                # ==========================================
                coords = v_item['center_coords'][0].cpu().numpy()
                cx, cy, cz = int(coords[0]), int(coords[1]), int(coords[2])

                M_anchor = np.zeros(EvalConfig.out_res, dtype=bool)
                M_anchor[cx, :, :] = True
                M_anchor[:, cy, :] = True
                M_anchor[:, :, cz] = True
                mask_purged = ~M_anchor  # 信息剥夺盲区

                # 格式转换用于 HD95 结算
                gt_b = torch.from_numpy(gt_mask_np).unsqueeze(0).unsqueeze(0).cuda()
                pred_b = torch.from_numpy(pred_mask_np).unsqueeze(0).unsqueeze(0).cuda()
                oracle_b = torch.from_numpy(oracle_aligned_np).unsqueeze(0).unsqueeze(0).cuda()

                # ==========================================
                # 🔴 绝对隔离算分结算
                # ==========================================
                o_iso_dice = compute_isolated_dice(oracle_aligned_np, gt_mask_np, mask_purged)
                p_iso_dice = compute_isolated_dice(pred_mask_np, gt_mask_np, mask_purged)

                o_glo_hd95 = safe_global_hd95(oracle_b, gt_b)
                p_glo_hd95 = safe_global_hd95(pred_b, gt_b)

                metrics['oracle_iso_dice'].append(o_iso_dice)
                metrics['pred_iso_dice'].append(p_iso_dice)
                metrics['oracle_glo_hd95'].append(o_glo_hd95)
                metrics['pred_glo_hd95'].append(p_glo_hd95)

                logger.info(f"[{v_item['name'][0]}] Iso_Dice (Oracle | {EvalConfig.model_type.upper()}): {o_iso_dice:.3f} | {p_iso_dice:.3f} --- Glo_HD95: {o_glo_hd95:.2f} | {p_glo_hd95:.2f}mm")

                if EvalConfig.save_vis:
                    vis_save_path = os.path.join(save_dir, 'vis', f"{v_item['name'][0]}_pred_{p_iso_dice:.3f}.png")
                    gt_img_np = v_item['target_image'][0, 0].cpu().numpy()
                    save_visualization_3view(
                        img_np=gt_img_np, prior_mask=prior_mask_np, aligned_prior=oracle_aligned_np,
                        gt_mask=gt_mask_np, pred_mask=pred_mask_np, save_path=vis_save_path,
                        case_name=v_item['name'][0], epoch=EvalConfig.model_type.upper()
                    )

    logger.info("\n" + "="*70)
    logger.info(f"🔥 FORENSIC AUTOPSY SUMMARY: {EvalConfig.model_type.upper()} 🔥")
    logger.info("="*70)
    logger.info(f" [Topology] Isolated Global Dice : Oracle {np.nanmean(metrics['oracle_iso_dice']):.4f} -> Pred {np.nanmean(metrics['pred_iso_dice']):.4f} ± {np.nanstd(metrics['pred_iso_dice']):.4f}")
    logger.info(f" [Geometry] Global HD95 (mm)     : Oracle {np.nanmean(metrics['oracle_glo_hd95']):.2f} -> Pred {np.nanmean(metrics['pred_glo_hd95']):.2f} ± {np.nanstd(metrics['pred_glo_hd95']):.2f}")
    logger.info(f" [Temporal] Inference Time       : {np.mean(metrics['inference_time']):.4f} s/vol")
    logger.info("="*70)