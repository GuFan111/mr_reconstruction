# test.py

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
import nibabel as nib

# 引入 MONAI 的评估度量工具
from monai.metrics import compute_hausdorff_distance, compute_dice, compute_average_surface_distance
from scipy.ndimage import center_of_mass, shift

from dataset import Prostate_Dataset
from models.model import DIF_Net
# 🟢 引入 Baseline 模型
from models.baseline_models import Baseline_3DUNet, Baseline_SwinUNETR
from utils import compute_com_error, save_visualization_3view

# ==========================================
#  🟢 全量程测试配置区域 (Test-Set Config)
# ==========================================
class EvalConfig:
    # 🔴 模型切换开关: 'difnet', '3dunet', 'swin_unetr'
    model_type = 'difnet'

    name = f'prostate_test_{model_type}'
    data_root = r'/root/autodl-tmp/Proj/data/prostate_158_128'

    # 🔴 请根据 model_type 填入对应的最佳权重路径
    checkpoint_path = r'/root/autodl-tmp/Proj/code/logs/prostate_new_2/model_best.pth'

    gpu_id = 0
    num_views = 3
    out_res = (128, 128, 128)
    combine = 'attention'
    save_vis = True  # 导出 2D 轮廓叠加图
    save_nii = True  # 导出 NIfTI 格式

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='w') # Eval 是覆盖写入 'w'
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

def save_nifti(array_np, save_path, is_mask=True):
    dtype = np.uint8 if is_mask else np.float32
    nii_img = nib.Nifti1Image(array_np.astype(dtype), np.eye(4))
    nib.save(nii_img, save_path)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(EvalConfig.gpu_id)
    save_dir = f'./logs/{EvalConfig.name}'
    os.makedirs(save_dir, exist_ok=True)
    if EvalConfig.save_vis:
        os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)
    if EvalConfig.save_nii:
        os.makedirs(os.path.join(save_dir, 'vis_nii'), exist_ok=True)

    logger = setup_logger(os.path.join(save_dir, 'test_log.txt'))
    logger.info(f"Start Evaluation on entire TEST Set...")
    logger.info(f"Selected Model: {EvalConfig.model_type.upper()}")
    logger.info(f"Loading checkpoint from: {EvalConfig.checkpoint_path}")

    # ==========================================
    # 🔴 关键修改：将读取目录和 split 全部指向 TEST
    # ==========================================
    test_dst = Prostate_Dataset(data_root=os.path.join(EvalConfig.data_root, 'test', 'image'),
                                label_root=os.path.join(EvalConfig.data_root, 'test', 'label'),
                                split='test', npoint=50000, out_res=EvalConfig.out_res)
    test_loader = DataLoader(test_dst, batch_size=1, shuffle=False)

    # ==========================================
    # 🔴 模型路由实例化
    # ==========================================
    if EvalConfig.model_type == 'difnet':
        model = DIF_Net(num_views=EvalConfig.num_views, combine=EvalConfig.combine).cuda()
    elif EvalConfig.model_type == '3dunet':
        model = Baseline_3DUNet().cuda()
    elif EvalConfig.model_type == 'swin_unetr':
        model = Baseline_SwinUNETR().cuda()
    else:
        raise ValueError(f"Unknown model_type: {EvalConfig.model_type}")

    if not os.path.exists(EvalConfig.checkpoint_path):
        logger.error(f"Checkpoint not found: {EvalConfig.checkpoint_path}")
        exit()

    checkpoint = torch.load(EvalConfig.checkpoint_path, map_location='cuda')
    # 处理 Baseline 的权重前缀 (部分框架可能会带 module.)
    if 'net.weight' in list(checkpoint.keys())[0] and EvalConfig.model_type != 'difnet':
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info(f"Model loaded successfully. Total TEST samples: {len(test_dst)}")

    metrics = {
        'com_init': [], 'com_pred': [],
        'dice_init': [], 'dice_align': [], 'dice_pred': [],
        'hd95_init': [], 'hd95_pred': [],
        'assd_init': [], 'assd_pred': [],
        'vol_diff_ratio': [],
        'inference_time': []
    }

    with torch.no_grad():
        with tqdm(test_loader, desc='Testing', ncols=100) as pbar:
            for i, v_item in enumerate(pbar):
                for key in v_item.keys():
                    if key not in ['name']: v_item[key] = v_item[key].float().cuda(non_blocking=True)

                torch.cuda.synchronize()
                t_start = time.time()

                # ==========================================
                # 🔴 前向推理路由
                # ==========================================
                if EvalConfig.model_type == 'difnet':
                    pred_logits = model(v_item, is_eval=True, eval_npoint=50000)
                else:
                    # Baseline 的提取逻辑
                    full_target_img = v_item['target_image'].float()
                    full_prior_img = v_item['prior_image'].float()
                    prior_mask_tensor = v_item['prior_mask'].float()
                    coords = v_item['center_coords']

                    # 使用自动混合精度加速基线推理
                    with torch.cuda.amp.autocast():
                        pred_logits = model(full_target_img, full_prior_img, prior_mask_tensor, coords)

                torch.cuda.synchronize()
                t_end = time.time()
                metrics['inference_time'].append(t_end - t_start)

                res = EvalConfig.out_res
                pred_probs = torch.sigmoid(pred_logits)
                pred_mask_tensor = (pred_probs > 0.5).float().view(1, 1, res[0], res[1], res[2])
                gt_mask_tensor = v_item['p_gt'].view(1, 1, res[0], res[1], res[2])
                prior_mask_tensor = v_item['prior_mask']

                pred_mask_np = pred_mask_tensor[0, 0].cpu().numpy()
                gt_mask_np = gt_mask_tensor[0, 0].cpu().numpy()
                prior_mask_np = prior_mask_tensor[0, 0].cpu().numpy()

                try:
                    com_gt = np.array(center_of_mass(gt_mask_np))
                    com_prior = np.array(center_of_mass(prior_mask_np))
                    shift_vec = com_gt - com_prior
                    aligned_prior_np = shift(prior_mask_np, shift_vec, order=0)
                    aligned_prior_tensor = torch.from_numpy(aligned_prior_np).unsqueeze(0).unsqueeze(0).cuda()
                except:
                    aligned_prior_tensor = prior_mask_tensor
                    aligned_prior_np = prior_mask_np

                init_com_err = compute_com_error(gt_mask_np, prior_mask_np, spacing=(1.0, 1.0, 1.0))
                recon_com_err = compute_com_error(gt_mask_np, pred_mask_np, spacing=(1.0, 1.0, 1.0))
                metrics['com_init'].append(init_com_err); metrics['com_pred'].append(recon_com_err)

                gt_b, pred_b, prior_b = gt_mask_tensor, pred_mask_tensor, prior_mask_tensor
                i_dice = compute_dice(prior_b, gt_b, include_background=False).item()
                a_dice = compute_dice(aligned_prior_tensor, gt_b, include_background=False).item()
                r_dice = compute_dice(pred_b, gt_b, include_background=False).item()
                metrics['dice_init'].append(i_dice); metrics['dice_align'].append(a_dice); metrics['dice_pred'].append(r_dice)

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

                if EvalConfig.save_vis:
                    vis_save_path = os.path.join(save_dir, 'vis', f"{v_item['name'][0]}_pred_{r_dice:.3f}.png")
                    gt_img_np = v_item['target_image'][0, 0].cpu().numpy()
                    save_visualization_3view(
                        img_np=gt_img_np, prior_mask=prior_mask_np, aligned_prior=aligned_prior_np,
                        gt_mask=gt_mask_np, pred_mask=pred_mask_np, save_path=vis_save_path,
                        case_name=v_item['name'][0], epoch="Test"
                    )

                if EvalConfig.save_nii:
                    case_id = v_item['name'][0]
                    nii_dir = os.path.join(save_dir, 'vis_nii')
                    save_nifti(gt_img_np, os.path.join(nii_dir, f"{case_id}_IMG.nii.gz"), is_mask=False)
                    save_nifti(gt_mask_np, os.path.join(nii_dir, f"{case_id}_GT.nii.gz"))
                    save_nifti(prior_mask_np, os.path.join(nii_dir, f"{case_id}_PRIOR.nii.gz"))
                    save_nifti(pred_mask_np, os.path.join(nii_dir, f"{case_id}_PRED_{r_dice:.3f}.nii.gz"))

    logger.info("\n" + "="*50)
    logger.info(f"FINAL TEST RESULTS: {EvalConfig.model_type.upper()}")
    logger.info("="*50)
    logger.info(f" - CoM Error (mm)      : Init {np.mean(metrics['com_init']):.2f} ± {np.std(metrics['com_init']):.2f} -> Pred {np.mean(metrics['com_pred']):.2f} ± {np.std(metrics['com_pred']):.2f}")
    logger.info(f" - Dice Score          : Init {np.mean(metrics['dice_init']):.4f} | Aligned {np.mean(metrics['dice_align']):.4f} | Pred {np.mean(metrics['dice_pred']):.4f} ± {np.std(metrics['dice_pred']):.4f}")
    logger.info(f" - HD95 (mm)           : Init {np.mean(metrics['hd95_init']):.2f} -> Pred {np.mean(metrics['hd95_pred']):.2f} ± {np.std(metrics['hd95_pred']):.2f}")
    logger.info(f" - ASSD (mm)           : Init {np.mean(metrics['assd_init']):.2f} -> Pred {np.mean(metrics['assd_pred']):.2f} ± {np.std(metrics['assd_pred']):.2f}")
    logger.info(f" - Volume Diff Ratio   : {np.mean(metrics['vol_diff_ratio']):.2f}% ± {np.std(metrics['vol_diff_ratio']):.2f}%")
    logger.info(f" - Inference Time      : {np.mean(metrics['inference_time']):.3f} s/vol")
    logger.info("="*50)