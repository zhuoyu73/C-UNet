import numpy as np

def compute_mean_std(img):
    return np.mean(img), np.std(img)

#mean_pred, std_pred = compute_mean_std(pred_artifact)
#mean_true, std_true = compute_mean_std(true_artifact)

def compute_residuals(pred_artifact, true_artifact, 
                      pred_clean, true_clean, 
                      true_lowrank, true_full):
    res1 = pred_artifact - true_artifact
    res2 = pred_clean - true_clean
    res3 = (true_lowrank + pred_artifact) - true_full
    return res1, res2, res3

from sklearn.metrics import mean_squared_error

def compute_nrmse(pred, true):
    mse = mean_squared_error(true.ravel(), pred.ravel())
    rmse = np.sqrt(mse)
    denom = np.max(true) - np.min(true)
    return rmse / denom

from skimage.metrics import structural_similarity as ssim

def compute_ssim(pred, true):
    ssim_val, _ = ssim(true, pred, full=True, data_range=true.max()-true.min())
    return ssim_val

from scipy.stats import pearsonr

def compute_corr(pred, true):
    pred_flat = pred.ravel()
    true_flat = true.ravel()
    corr, _ = pearsonr(pred_flat, true_flat)
    return corr

from scipy.ndimage import gaussian_filter

def compute_corr_gaussian(pred, true, sigma=1):
    pred_filt = gaussian_filter(pred, sigma=sigma)
    true_filt = gaussian_filter(true, sigma=sigma)
    return compute_corr(pred_filt, true_filt)

def compute_mse(pred, true):
    return mean_squared_error(true.ravel(), pred.ravel())

def compute_normalized_mse(pred, true):
    mse = compute_mse(pred, true)
    return mse / np.var(true)


def evaluate_metrics(pred_artifact, true_artifact, 
                     pred_clean, true_clean, 
                     true_lowrank, true_full):
    results = {}

    # 1. Mean & Std
    results['mean_pred_artifact'], results['std_pred_artifact'] = compute_mean_std(pred_artifact)
    results['mean_true_artifact'], results['std_true_artifact'] = compute_mean_std(true_artifact)

    # 2. Residual maps
    res1, res2, res3 = compute_residuals(pred_artifact, true_artifact, 
                                         pred_clean, true_clean, 
                                         true_lowrank, true_full)
    results['residual_maps'] = (res1, res2, res3)

    # 3. Performance
    results['NRMSE'] = compute_nrmse(pred_clean, true_full)
    results['SSIM'] = compute_ssim(pred_clean, true_full)
    results['corr_before'] = compute_corr(pred_clean, true_full)
    results['corr_after'] = compute_corr_gaussian(pred_clean, true_full, sigma=1)

    # 4. MSE
    results['MSE_artifact'] = compute_mse(pred_artifact, true_artifact)
    results['MSE_clean'] = compute_mse(pred_clean, true_full)
    results['NMSE_artifact'] = compute_normalized_mse(pred_artifact, true_artifact)
    results['NMSE_clean'] = compute_normalized_mse(pred_clean, true_full)

    return results



from scipy.io import loadmat
import h5py


true_lowrank = loadmat("/mnt/external/zhuoyu/fully+osci/250326_ZS/250326_ZS_B1000/mbmre_both/img_US.mat")['img']
true_full = loadmat("/mnt/external/zhuoyu/fully+osci/250326_ZS/250326_ZS_B1000/mbmre/img.mat")['img']
pred_clean = loadmat("/home/zhuoyu/Desktop/C-UNet/250404_ArtifactNet/250731_14_results/250326_ZS/artifact_pred.mat")['img']


assert true_lowrank.shape == true_full.shape == pred_clean.shape

true_artifact = true_lowrank - true_full
true_clean = true_full
pred_artifact = true_lowrank - pred_clean

results = evaluate_metrics(pred_artifact, true_artifact,
                           pred_clean, true_clean,
                           true_lowrank, true_full)

for k,v in results.items():
    if k != "residual_maps":
        print(k, ":", v)
