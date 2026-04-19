# eval_qin_robustness_full_metrics.py
# 显著性测试

import os
import warnings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore') # 忽略小样本下Wilcoxon可能的Tie警告

import time
import logging
import numpy as np
import pandas as pd
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
# 🟢 临床极端感知破坏仿真器
# ==========================================
def academic_mask_noise_generator(mask_2d, severity_level):
    if severity_level == 0: return mask_2d.copy().astype(np.float32)
    mask_noisy = mask_2d.copy().astype(float)
    
    def apply_morph(m, s):
        action = np.random.choice(['dilate', 'erode'])
        return ndimage.binary_dilation(m, iterations=s) if action == 'dilate' else ndimage.binary_erosion(m, iterations=s)
        
    def apply_elastic(m, s):
        shape = m.shape
        alpha = s * 25.0  
        sigma = 4.0       
        dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        return ndimage.map_coordinates(m, indices, order=0, mode='constant').reshape(shape) > 0.5
        
    def apply_dropout(m, s):
        m_out = m.copy()
        cx, cy = np.random.randint(40, 88, 2)
        r = int(s * 2.5)  
        y, x = np.ogrid[:128, :128]
        hole = (x - cx)**2 + (y - cy)**2 <= r**2
        m_out[hole] = 0
        if s >= 2:
            cx2, cy2 = np.random.randint(20, 108, 2)
            r2 = int(s * 2.0)
            m_out[(x - cx2)**2 + (y - cy2)**2 <= r2**2] = 1
        return m_out

    # Mixed Noise
    mask_noisy = apply_elastic(mask_noisy, severity_level)
    mask_noisy = apply_morph(mask_noisy, max(1, severity_level // 2))
    mask_noisy = apply_dropout(mask_noisy, max(1, severity_level - 1))
    return mask_noisy.astype(np.float32)

# ==========================================
# 🟢 狄利克雷拉普拉斯算子
# ==========================================
def dirichlet_harmonic_diffusion(prior_3d, t2d, cx, cy, cz, iters=50):
    t2d = torch.where(t2d > 0.5, 10.0, -10.0).float()
    b = torch.zeros_like(prior_3d, dtype=torch.bool)
    b[:,:,:,:,cz] = b[:,:,:,cy,:] = b[:,:,cx,:,:] = True
    r = torch.zeros_like(prior_3d)
    r[:,:,:,:,cz], r[:,:,:,cy,:], r[:,:,cx,:,:] = t2d[0]-prior_3d[:,:,:,:,cz], t2d[1]-prior_3d[:,:,:,cy,:], t2d[2]-prior_3d[:,:,cx,:,:]
    df = torch.zeros_like(prior_3d)
    df[b] = r[b]
    k = torch.ones(1,1,3,3,3, device=prior_3d.device); k[0,0,1,1,1] = 0; k /= k.sum()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(iters):
            df = torch.where(b, r, F.conv3d(df, k, padding=1))
    return torch.sigmoid(prior_3d + df)

# ==========================================
# 🟢 杂项与评价工具
# ==========================================
def keep_lcc(m):
    b = m > 0.5; 
    if not b.any(): return m
    lbl, n = ndimage.label(b)
    return (lbl == np.bincount(lbl.ravel())[1:].argmax() + 1).astype(m.dtype)

def smooth_binary_mask(mask_np, sigma=1.0, threshold=0.5):
    return (ndimage.gaussian_filter(mask_np.astype(np.float32), sigma=sigma) > threshold).astype(np.uint8)

def compute_band_dice(p, g, b):
    inter = np.logical_and(p[b]>0.5, g[b]>0.5).sum()
    union = (p[b]>0.5).sum() + (g[b]>0.5).sum()
    return np.nan if union == 0 else 2.0 * inter / union

def safe_global_hd95(p_tensor, g_tensor):
    if p_tensor.sum() == 0 or g_tensor.sum() == 0: return 99.0
    try: return compute_hausdorff_distance(p_tensor, g_tensor, include_background=False, percentile=95).item()
    except: return 99.0

class QinBlindTestDataset(Dataset):
    def __init__(self, data_root):
        self.patient_dirs = sorted([os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')])
    def __len__(self): return len(self.patient_dirs)
    def __getitem__(self, idx):
        p_dir = self.patient_dirs[idx]
        return {'prior_image': np.load(os.path.join(p_dir, 'prior_image.npy')), 'prior_mask': np.load(os.path.join(p_dir, 'prior_mask.npy')),
                'target_image': np.load(os.path.join(p_dir, 'target_image.npy')), 'target_mask': np.load(os.path.join(p_dir, 'target_mask.npy'))}

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

def fmt_sig(val, p_val, is_hd95=False):
    if np.isnan(p_val) or p_val == 1.0: 
        return f"{val:.4f}  " if not is_hd95 else f"{val:.2f}  "
    stars = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "  "
    return f"{val:.4f}{stars}" if not is_hd95 else f"{val:.2f}{stars}"

# ==========================================
# 🟢 主控程序
# ==========================================
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_root = '/root/autodl-tmp/Proj/data/qin_testset_npy'
    save_dir = './logs/qin_robustness_metrics_full'
    os.makedirs(save_dir, exist_ok=True)
    
    logger = setup_logger(os.path.join(save_dir, 'full_metrics_with_pvalue_log.txt'))
    logger.info("🚀 [STRESS TEST FULL METRICS] 启动全量衰减数据收集引擎 (附带 P-Value 显著性追踪)...")

    models_dict = {
        'ATT_Laplace': DIF_Net(num_views=3, combine='attention').cuda(),
        'MLP_Laplace': DIF_Net(num_views=3, combine='mlp').cuda(),
        'SWIN_Laplace': Baseline_SwinUNETR().cuda()
    }
    weights = {
        'ATT_Laplace': '/root/autodl-tmp/Proj/code/logs/prostate_4_8_attention/model_best.pth',
        'MLP_Laplace': '/root/autodl-tmp/Proj/code/logs/prostate_4_8_mlp/model_best.pth',
        'SWIN_Laplace': '/root/autodl-tmp/Proj/code/logs/baseline_swin_unetr_sparse_amp_4_8/model_best.pth'
    }
    
    for m, net in models_dict.items():
        net.load_state_dict(torch.load(weights[m], map_location='cuda'), strict=False)
        net.eval()

    val_dst = QinBlindTestDataset(data_root)
    geo = OrthogonalGeometry()
    
    severities = [0, 1, 2, 3, 4, 5]
    final_csv_data = []

    for sev in severities:
        logger.info(f"\n" + "="*85)
        logger.info(f"☣️  Running Severity Level {sev} (Mixed Noise)...")
        
        metrics = {m: {'glo':[], 'b13':[], 'b47':[], 'bgt7':[], 'hd95':[]} for m in models_dict.keys()}
        noisy_input_dices = []
        
        for idx in tqdm(range(len(val_dst)), ncols=80, desc=f"Sev-{sev}"):
            item = val_dst[idx]
            pi, pm, ti, tm = item['prior_image'], item['prior_mask'], item['target_image'], item['target_mask']
            nz_p = np.argwhere(pm > 0); px, py, pz = nz_p.mean(axis=0).astype(int) if len(nz_p)>0 else (64,64,64)
            nz_t = np.argwhere(tm > 0); fx, fy, fz = nz_t.mean(axis=0).astype(int) if len(nz_t)>0 else (64,64,64)
            
            c_ti = ndimage.shift(ti, (64-fx, 64-fy, 64-fz), order=1)
            c_tm = ndimage.shift(tm, (64-fx, 64-fy, 64-fz), order=0)
            c_pi = ndimage.shift(pi, (64-px, 64-py, 64-pz), order=1)
            c_pm = ndimage.shift(pm, (64-px, 64-py, 64-pz), order=0)

            # 1. 制作公共毒药
            gt_s = np.stack([c_tm[:,:,64], c_tm[:,64,:], c_tm[64,:,:]])
            noisy_s, tmp_dices = np.zeros_like(gt_s), []
            np.random.seed(idx * 100) 
            for i in range(3): 
                noisy_s[i] = academic_mask_noise_generator(gt_s[i], sev)
                inter = np.sum(noisy_s[i] * gt_s[i])
                tmp_dices.append(2.0 * inter / (np.sum(noisy_s[i]) + np.sum(gt_s[i]) + 1e-5))
            noisy_input_dices.append(np.mean(tmp_dices))
            t2d_tensor = torch.from_numpy(noisy_s).float().cuda()

            # 2. 划定频带掩膜与GT Tensor
            M = np.zeros((128,128,128), dtype=bool); M[fx,:,:] = M[:,fy,:] = M[:,:,fz] = True
            m_purged = ~M
            dist = ndimage.distance_transform_edt(m_purged)
            m_b13 = (dist > 0) & (dist <= 3); m_b47 = (dist > 3) & (dist <= 7); m_bgt7 = (dist > 7)
            gt_tensor = torch.from_numpy(smooth_binary_mask(tm)).view(1,1,128,128,128).float().cuda()

            # 3. 多模型公平竞技
            res = 128; grid = np.mgrid[:res,:res,:res].reshape(3,-1).transpose(1,0)
            pts = ((grid.astype(np.float32) / 127.0) - 0.5) * 2
            p_pts = np.stack([geo.project(pts, 0), geo.project(pts, 1), geo.project(pts, 2)], axis=0)
            
            for m_name, model in models_dict.items():
                with torch.no_grad():
                    # 🟢 修正点：正确的模型字符串判断逻辑
                    if 'ATT' in m_name or 'MLP' in m_name:
                        p_in = np.zeros((3,1,128,128), dtype=np.float32); p_in[0,0]=c_ti[:,:,64]; p_in[1,0]=c_ti[:,64,:]; p_in[2,0]=c_ti[64,:,:]
                        pi_in = np.zeros((3,1,128,128), dtype=np.float32); pi_in[0,0]=c_pi[:,:,64]; pi_in[1,0]=c_pi[:,64,:]; pi_in[2,0]=c_pi[64,:,:]
                        out = model({'projs': torch.from_numpy(p_in).unsqueeze(0).cuda(), 'prior_projs': torch.from_numpy(pi_in).unsqueeze(0).cuda(),
                                     'prior_mask': torch.from_numpy(c_pm).view(1,1,128,128,128).float().cuda(),
                                     'points': torch.from_numpy(pts).unsqueeze(0).cuda(), 'proj_points': torch.from_numpy(p_pts).unsqueeze(0).cuda()}, is_eval=True, eval_npoint=50000)
                        f_prob = dirichlet_harmonic_diffusion(out.view(1,1,128,128,128), t2d_tensor, 64, 64, 64, 50)
                        pred = ndimage.shift((f_prob.cpu().numpy().squeeze()>0.5).astype(np.uint8), (fx-64, fy-64, fz-64), order=0)
                    else:
                        with torch.amp.autocast('cuda'):
                            out = model(torch.from_numpy(ti).view(1,1,128,128,128).float().cuda(), 
                                        torch.from_numpy(ndimage.shift(pi, (fx-px, fy-py, fz-pz), order=1)).view(1,1,128,128,128).float().cuda(), 
                                        torch.from_numpy(ndimage.shift(pm, (fx-px, fy-py, fz-pz), order=0)).view(1,1,128,128,128).float().cuda(), 
                                        torch.tensor([[fx,fy,fz]], dtype=torch.long).cuda())
                        c_out = torch.from_numpy(ndimage.shift(out[0,0].float().cpu().numpy(), (64-fx, 64-fy, 64-fz), order=1)).cuda().view(1,1,128,128,128)
                        f_prob = dirichlet_harmonic_diffusion(c_out, t2d_tensor, 64, 64, 64, 50)
                        pred = ndimage.shift((f_prob.cpu().numpy().squeeze()>0.5).astype(np.uint8), (fx-64, fy-64, fz-64), order=0)
                
                # 算分
                pred_lcc = keep_lcc(pred)
                pred_tensor = torch.from_numpy(smooth_binary_mask(pred_lcc)).view(1,1,128,128,128).float().cuda()
                
                metrics[m_name]['glo'].append(compute_band_dice(pred_lcc, tm, m_purged))
                metrics[m_name]['b13'].append(compute_band_dice(pred_lcc, tm, m_b13))
                metrics[m_name]['b47'].append(compute_band_dice(pred_lcc, tm, m_b47))
                metrics[m_name]['bgt7'].append(compute_band_dice(pred_lcc, tm, m_bgt7))
                metrics[m_name]['hd95'].append(safe_global_hd95(pred_tensor, gt_tensor))

        # ---------------------------------
        # 4. 计算双侧 P-Value 与 收集数据
        # ---------------------------------
        mean_noise = np.mean(noisy_input_dices)
        prop_glo = np.array(metrics['ATT_Laplace']['glo'])
        prop_bgt7 = np.array(metrics['ATT_Laplace']['bgt7'])
        prop_hd95 = np.array(metrics['ATT_Laplace']['hd95'])

        p_vals = {'ATT_Laplace': {'glo': 1.0, 'bgt7': 1.0, 'hd95': 1.0}}
        
        for base in ['MLP_Laplace', 'SWIN_Laplace']:
            base_glo = np.array(metrics[base]['glo'])
            base_bgt7 = np.array(metrics[base]['bgt7'])
            base_hd95 = np.array(metrics[base]['hd95'])
            
            # 双侧检验
            _, p_glo = wilcoxon(prop_glo, base_glo, alternative='two-sided')
            _, p_bgt7 = wilcoxon(prop_bgt7, base_bgt7, alternative='two-sided')
            _, p_hd95 = wilcoxon(prop_hd95, base_hd95, alternative='two-sided')
            
            p_vals[base] = {'glo': p_glo, 'bgt7': p_bgt7, 'hd95': p_hd95}

        # 写入大表数据
        for m_name in models_dict.keys():
            record = {
                'Severity': sev,
                'Input_2D_Dice': mean_noise,
                'Model': m_name,
                'Global_Purged': np.nanmean(metrics[m_name]['glo']),
                'Band_1_3v': np.nanmean(metrics[m_name]['b13']),
                'Band_4_7v': np.nanmean(metrics[m_name]['b47']),
                'Band_gt_7v': np.nanmean(metrics[m_name]['bgt7']),
                'Global_HD95': np.nanmean(metrics[m_name]['hd95']),
                'P_val_Global': p_vals[m_name]['glo'],
                'P_val_gt7v': p_vals[m_name]['bgt7'],
                'P_val_HD95': p_vals[m_name]['hd95']
            }
            final_csv_data.append(record)
            
    # ==========================================
    # 🏁 终极输出：Markdown 表格与 CSV
    # ==========================================
    df = pd.DataFrame(final_csv_data)
    csv_path = os.path.join(save_dir, "robustness_all_metrics_with_pvalues.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info("\n" + "🔥"*45)
    logger.info("    FULL DEGRADATION MATRIX WITH SIGNIFICANCE (* p<0.05, ** p<0.01)")
    logger.info("🔥"*45)
    
    header = f"| {'Sev':<3} | {'Model':<14} | {'Input2D':<7} | {'Global':<9} | {'1-3v':<7} | {'4-7v':<7} | {'>7v':<9} | {'HD95':<8} |"
    logger.info(header)
    logger.info("|" + "-"*85 + "|")
    
    for sev in severities:
        sev_df = df[df['Severity'] == sev]
        for _, row in sev_df.iterrows():
            m_name = row['Model']
            str_glo = fmt_sig(row['Global_Purged'], row['P_val_Global'])
            str_gt7 = fmt_sig(row['Band_gt_7v'], row['P_val_gt7v'])
            str_hd95 = fmt_sig(row['Global_HD95'], row['P_val_HD95'], is_hd95=True)
            
            logger.info(f"| {row['Severity']:<3} | {m_name:<14} | {row['Input_2D_Dice']:.4f} | "
                        f"{str_glo:<9} | {row['Band_1_3v']:.4f} | {row['Band_4_7v']:.4f} | "
                        f"{str_gt7:<9} | {str_hd95:<8} |")
        if sev != severities[-1]:
            logger.info("|" + "-"*85 + "|")
            
    logger.info("="*89)
    logger.info(f"✅ 全量数据及 P-value 已保存至: {csv_path}")
    logger.info("💡 结论指引：观察 >7v 列的星号 (*) 如何随着 Severity 增加而出现并加深，这是最完美的鲁棒性证明！")