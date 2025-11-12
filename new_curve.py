import os
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
from scipy.stats import ttest_ind, mannwhitneyu
import numpy as np

def resample_curve(points, M=400):
    P = np.asarray(points)
    x, y = P[:,0], P[:,1]
    dx, dy = np.diff(x), np.diff(y)
    s = np.r_[0.0, np.cumsum(np.hypot(dx, dy))]
    t = s / s[-1]
    splx = splrep(t, x, k=3, s=0)
    sply = splrep(t, y, k=3, s=0)
    t_dense = np.linspace(0,1,M)
    xM = splev(t_dense, splx)
    yM = splev(t_dense, sply)
    return np.c_[xM, yM]

def robust_sigma_from_trials(trials_points, M=400):
    curves = [resample_curve(P, M=M) for P in trials_points]
    C = np.stack(curves, axis=1)   # (M, T, 2)
    mad_x = 1.4826 * np.median(np.abs(C[:,:,0] - np.median(C[:,:,0],1,keepdims=True)), axis=1)
    mad_y = 1.4826 * np.median(np.abs(C[:,:,1] - np.median(C[:,:,1],1,keepdims=True)), axis=1)
    sigma_j = np.sqrt(mad_x**2 + mad_y**2)
    return float(np.clip(np.median(sigma_j), 1.0, 5.0)) 

def compute_tac_mac(points, sigma_px, n_eval=500):
    points = np.array(points)
    x, y = points[:,0], points[:,1]
    dx = np.diff(x); dy = np.diff(y)
    seg_lengths = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0], np.cumsum(seg_lengths)))
    t = s / s[-1]

    smooth_val = len(points) * (sigma_px**2)
    splx = splrep(t, x, k=3, s=smooth_val)
    sply = splrep(t, y, k=3, s=smooth_val)

    tnew = np.linspace(0,1,n_eval)
    x1 = splev(tnew, splx, der=1)
    x2 = splev(tnew, splx, der=2)
    y1 = splev(tnew, sply, der=1)
    y2 = splev(tnew, sply, der=2)
    kappa = np.abs(x1*y2 - y1*x2) / (x1**2 + y1**2)**1.5

    dt = tnew[1] - tnew[0]
    ds = np.sqrt(x1**2 + y1**2) * dt

    tac = np.sum(kappa * ds)
    arc_length = np.sum(ds)
    mac = tac / arc_length
    return tac, mac

results = []
base = "results"


"""
This loop is set up for triplicate analysis of preannotated images using the code in picker.py
This may need to be modified for specific use cases
"""
for trial in ["trial_1", "trial_2", "trial_3"]:
    for style in ["old", "new"]:
        folder = os.path.join(base, trial, style)
        for fname in os.listdir(folder):
            if fname.endswith(".csv"):
                side = "left" if "left" in fname else "right"
                patient = fname.split("_")[0].replace("Picture","")
                df = np.loadtxt(os.path.join(folder,fname), delimiter=",")
                results.append({
                    "trial": trial,
                    "style": style,
                    "patient": patient,
                    "side": side,
                    "points": df
                })

df_all = pd.DataFrame(results)


'''
Do the actual analysis by computing dynamic sigma based on pixel variance across trials
'''

sigmas = {}
for key, group in df_all.groupby(["patient","style","side"]):
    pts_trials = group["points"].tolist()
    sigma_px = robust_sigma_from_trials(pts_trials)
    sigmas[key] = sigma_px

trial_records = []
for i, row in df_all.iterrows():
    key = (row["patient"], row["style"], row["side"])
    sigma_px = sigmas[key]
    tac, mac = compute_tac_mac(row["points"], sigma_px=sigma_px)
    trial_records.append({
        "patient": row["patient"],
        "style": row["style"],
        "side": row["side"],
        "trial": row["trial"],
        "sigma_px": sigma_px,
        "tac": tac,
        "mac": mac
    })

df_trials = pd.DataFrame(trial_records)



'''
Ugly but functional analysis code. Depending on structure of your data, ymmv
'''

df_summary = (df_trials
    .groupby(["patient","style","side"])
    .agg(mean_tac=("tac","mean"), std_tac=("tac","std"),
         mean_mac=("mac","mean"), std_mac=("mac","std"),
         sigma_px=("sigma_px","first"))   # same σ for all trials in group
    .reset_index()
)

df_trials.to_csv("output_trials.csv", index=False)
df_summary.to_csv("output_summary.csv", index=False)



def cohen_d(x, y):
    nx, ny = len(x), len(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    # pooled standard deviation
    sp = np.sqrt(((nx-1)*sx + (ny-1)*sy) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / sp

old_TAC = df_summary.loc[df_summary["style"]=="old", "mean_tac"].values
new_TAC = df_summary.loc[df_summary["style"]=="new", "mean_tac"].values

t_stat, p_val = ttest_ind(old_TAC, new_TAC, equal_var=False)

d_val = cohen_d(old_TAC, new_TAC)

print(f"TAC old vs new:")
print(f"  Old mean ± SD: {old_TAC.mean():.3f} ± {old_TAC.std(ddof=1):.3f}")
print(f"  New mean ± SD: {new_TAC.mean():.3f} ± {new_TAC.std(ddof=1):.3f}")
print(f"  Cohen’s d: {d_val:.3f}")
