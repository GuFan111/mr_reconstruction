# eval_qin_forensic_nolaplace_pvalue.py

import os
import warnings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import sys
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
from tqdm import tqdm
import nibabel as nib
from scipy.stats import wilcoxon

from monai.metrics import compute_hausdorff_distance
from torch.utils.data import Dataset

from models.model import DIF_Net
from models.baseline_models import Baseline_SwinUNETR
from dataset import OrthogonalGeometry

# ==========================================
# 🟢 杂项与评价工具
# ==========================================
def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

def keep_largest_connected_component_3d(mask_3d):
    mask_bool = mask_3d > 0.5
    if not mask_bool.any(): return mask_3d 
    labeled_mask, num_features = ndimage.label(mask_bool)
    if num_features <= 1: return mask_3d
    return (labeled_mask == np.bincount(labeled_mask.ravel())[1:].argmax() + 1).astype(mask_3d.dtype)

def smooth_binary_mask(mask_np, sigma=1.0, threshold=0.5):
    return (ndimage.gaussian_filter(mask_np.astype(np.float32), sigma=sigma) > threshold).astype(np.uint8)

def compute_band_dice(pred_mask, gt_mask, band_mask):
    p, g = pred_mask[band_mask].astype(bool), gt_mask[band_mask].astype(bool)
    union = p.sum() + g.sum()
    return np.nan if union == 0 else 2.0 * np.logical_and(p, g).sum() / union

def safe_global_hd95(pred_tensor, gt_tensor):
    if pred_tensor.sum() == 0 or gt_tensor.sum() == 0: return 99.0
    try: return compute_hausdorff_distance(pred_tensor, gt_tensor, include_background=False, percentile=95).item()
    except: return 99.0

class QinBlindTestDataset(Dataset):
    def __init__(self, data_root):
        self.patient_dirs = sorted([os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')])
    def __len__(self): return len(self.patient_dirs)
    def __getitem__(self, idx):
        p_dir = self.patient_dirs[idx]
        return {'name': os.path.basename(p_dir),
                'prior_image': np.load(os.path.join(p_dir, 'prior_image.npy')), 'prior_mask': np.load(os.path.join(p_dir, 'prior_mask.npy')),
                'target_image': np.load(os.path.join(p_dir, 'target_image.npy')), 'target_mask': np.load(os.path.join(p_dir, 'target_mask.npy'))}

# ==========================================
# 🟢 纯网络评估配置
# ==========================================
class PureNetConfig:
    gpu_id = 0
    data_root = '/root/autodl-tmp/Proj/data/qin_testset_npy'
    
    proposed_model = 'difnet_att'
    baseline_models = ['difnet_mlp', 'swin_unetr']
    
    weights = {
        'difnet_att': '/root/autodl-tmp/Proj/code/logs/prostate_4_8_attention/model_best.pth',
        'difnet_mlp': '/root/autodl-tmp/Proj/code/logs/prostate_4_8_mlp/model_best.pth',
        'swin_unetr': '/root/autodl-tmp/Proj/code/logs/baseline_swin_unetr_sparse_amp_4_8/model_best.pth'
    }

# ==========================================
# 🟢 模块化推理函数 (NO LAPLACE)
# ==========================================
def evaluate_single_model(m_name, dataset, geo):
    if 'difnet' in m_name:
        combine_mode = 'attention' if 'att' in m_name else 'mlp'
        model = DIF_Net(num_views=3, combine=combine_mode).cuda()
    else:
        model = Baseline_SwinUNETR().cuda()
        
    model.load_state_dict(torch.load(PureNetConfig.weights[m_name], map_location='cuda'), strict=False)
    model.eval()

    metrics = {'glo': [], 'bgt7': [], 'hd95': [], 't_intra': []}

    for idx in tqdm(range(len(dataset)), ncols=80, desc=m_name.upper()):
        item = dataset[idx]
        pi, pm, ti, tm = item['prior_image'], item['prior_mask'], item['target_image'], item['target_mask']
        nz_p = np.argwhere(pm > 0); px, py, pz = nz_p.mean(axis=0).astype(int) if len(nz_p)>0 else (64,64,64)
        nz_t = np.argwhere(tm > 0); fx, fy, fz = nz_t.mean(axis=0).astype(int) if len(nz_t)>0 else (64,64,64)

        c_ti = ndimage.shift(ti, (64-fx, 64-fy, 64-fz), order=1)
        c_pi = ndimage.shift(pi, (64-px, 64-py, 64-pz), order=1)
        c_pm = ndimage.shift(pm, (64-px, 64-py, 64-pz), order=0)

        t1 = time.time()
        
        # ----------------------------------------------------
        # 🧪 纯网络推理 (直接 Sigmoid 截断，无扩散)
        # ----------------------------------------------------
        with torch.no_grad():
            if 'difnet' in m_name:
                p_in = np.zeros((3,1,128,128), dtype=np.float32); p_in[0,0]=c_ti[:,:,64]; p_in[1,0]=c_ti[:,64,:]; p_in[2,0]=c_ti[64,:,:]
                pi_in = np.zeros((3,1,128,128), dtype=np.float32); pi_in[0,0]=c_pi[:,:,64]; pi_in[1,0]=c_pi[:,64,:]; pi_in[2,0]=c_pi[64,:,:]
                
                res = 128; grid = np.mgrid[:res,:res,:res].reshape(3,-1).transpose(1,0)
                pts = ((grid.astype(np.float32) / 127.0) - 0.5) * 2
                p_pts = np.stack([geo.project(pts, 0), geo.project(pts, 1), geo.project(pts, 2)], axis=0)

                pred_logits = model({'projs': torch.from_numpy(p_in).unsqueeze(0).cuda(), 
                                     'prior_projs': torch.from_numpy(pi_in).unsqueeze(0).cuda(),
                                     'prior_mask': torch.from_numpy(c_pm).view(1,1,128,128,128).float().cuda(),
                                     'points': torch.from_numpy(pts).unsqueeze(0).cuda(), 
                                     'proj_points': torch.from_numpy(p_pts).unsqueeze(0).cuda()}, is_eval=True, eval_npoint=50000)
                
                # 直接 Sigmoid 转 numpy (NO LAPLACE)
                pred_mask_centered = (torch.sigmoid(pred_logits).view(128, 128, 128).cpu().numpy() > 0.5).astype(np.uint8)
                pred_mask_np = ndimage.shift(pred_mask_centered, (fx-64, fy-64, fz-64), order=0)

            else:
                with torch.amp.autocast('cuda'):
                    pred_logits = model(torch.from_numpy(ti).view(1,1,128,128,128).float().cuda(), 
                                        torch.from_numpy(ndimage.shift(pi, (fx-px, fy-py, fz-pz), order=1)).view(1,1,128,128,128).float().cuda(), 
                                        torch.from_numpy(ndimage.shift(pm, (fx-px, fy-py, fz-pz), order=0)).view(1,1,128,128,128).float().cuda(), 
                                        torch.tensor([[fx,fy,fz]], dtype=torch.long).cuda())
                pred_mask_np = (torch.sigmoid(pred_logits).cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
        
        torch.cuda.synchronize()
        metrics['t_intra'].append(time.time() - t1)

        # 算分
        pred_lcc = keep_largest_connected_component_3d(pred_mask_np)
        
        M = np.zeros((128,128,128), dtype=bool); M[fx,:,:] = M[:,fy,:] = M[:,:,fz] = True
        m_purged = ~M; dist = ndimage.distance_transform_edt(m_purged); m_bgt7 = (dist > 7)
        
        gt_tensor = torch.from_numpy(smooth_binary_mask(tm)).view(1,1,128,128,128).float().cuda()
        pred_tensor = torch.from_numpy(smooth_binary_mask(pred_lcc)).view(1,1,128,128,128).float().cuda()
        
        metrics['glo'].append(compute_band_dice(pred_lcc, tm, m_purged))
        metrics['bgt7'].append(compute_band_dice(pred_lcc, tm, m_bgt7))
        metrics['hd95'].append(safe_global_hd95(pred_tensor, gt_tensor))

    return metrics

# ==========================================
# 🟢 主控程序：收集并执行 P-Value 计算
# ==========================================
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(PureNetConfig.gpu_id)
    save_dir = f'./logs/qin_nolaplace_pvalue'
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(os.path.join(save_dir, 'pure_net_pvalue_log.txt'))
    
    logger.info("🚀 [PURE NETWORK ARCHITECTURE TEST] 启动纯净网络架构消融 (彻底禁用 Laplace)...")
    val_dst = QinBlindTestDataset(data_root=PureNetConfig.data_root)
    geo = OrthogonalGeometry()

    all_models = [PureNetConfig.proposed_model] + PureNetConfig.baseline_models
    all_results = {}

    for m_name in all_models:
        all_results[m_name] = evaluate_single_model(m_name, val_dst, geo)

    # 计算 双侧 Wilcoxon 检验
    logger.info("\n" + "="*85)
    logger.info("📊 PURE ARCHITECTURE SIGNIFICANCE (TWO-SIDED WILCOXON, NO LAPLACE)")
    logger.info("="*85)
    logger.info(f"Target Hypothesis: {PureNetConfig.proposed_model.upper()} vs Baselines.")
    logger.info("-" * 85)
    
    prop_glo = np.array(all_results[PureNetConfig.proposed_model]['glo'])
    prop_far = np.array(all_results[PureNetConfig.proposed_model]['bgt7'])
    prop_hd95 = np.array(all_results[PureNetConfig.proposed_model]['hd95'])
    prop_time = np.mean(all_results[PureNetConfig.proposed_model]['t_intra'])

    logger.info(f"PROPOSED [{PureNetConfig.proposed_model.upper()}] - Glo: {np.mean(prop_glo):.4f} | >7v: {np.mean(prop_far):.4f} | HD95: {np.mean(prop_hd95):.2f}mm | Latency: {prop_time*1000:.1f}ms")
    logger.info("-" * 85)

    for base in PureNetConfig.baseline_models:
        base_glo = np.array(all_results[base]['glo'])
        base_far = np.array(all_results[base]['bgt7'])
        base_hd95 = np.array(all_results[base]['hd95'])
        base_time = np.mean(all_results[base]['t_intra'])
        
        logger.info(f"BASELINE [{base.upper()}] - Glo: {np.mean(base_glo):.4f} | >7v: {np.mean(base_far):.4f} | HD95: {np.mean(base_hd95):.2f}mm | Latency: {base_time*1000:.1f}ms")

        # 🟢 严格的双侧检验
        _, p_glo = wilcoxon(prop_glo, base_glo, alternative='two-sided')
        _, p_far = wilcoxon(prop_far, base_far, alternative='two-sided')
        _, p_hd95 = wilcoxon(prop_hd95, base_hd95, alternative='two-sided')
        
        sig_glo = "★ SIGNIFICANT" if p_glo < 0.05 else "n.s."
        sig_far = "★ SIGNIFICANT" if p_far < 0.05 else "n.s."
        sig_hd95 = "★ SIGNIFICANT" if p_hd95 < 0.05 else "n.s."
        
        logger.info(f"📌 VS {base.upper()}:")
        logger.info(f"   - Global Purged Dice : P-value = {p_glo:.4f}  [{sig_glo}]")
        logger.info(f"   - Deep Void (>7v)    : P-value = {p_far:.4f}  [{sig_far}]")
        logger.info(f"   - Global HD95 (mm)   : P-value = {p_hd95:.4f}  [{sig_hd95}]")
        logger.info("-" * 85)

    logger.info("💡 结论指引：这组数据揭示了纯粹的网络（没有数学扩散加持）下，ATT 的路由能力与计算效率（Latency）！")