"""Extract statistics from Part 1 homography files."""
import scipy.io as sio
import os
import numpy as np

path = r"c:\Users\acach\OneDrive\Documents\PIV\Datasets\Taag\sequence_homographies"

files = sorted([f for f in os.listdir(path) if f.endswith('.mat') and f.startswith('homography')])

print("Frame | Raw Matches | Inliers | Ratio (%) | Error (px)")
print("-" * 55)

all_raw = []
all_inl = []
all_rat = []
all_err = []

for f in files:
    data = sio.loadmat(os.path.join(path, f))
    frame = f.split('_')[-1].replace('.mat', '')
    
    # Check if new stats exist
    if 'raw_matches' in data:
        raw = int(data['raw_matches'][0][0])
        inl = int(data['ransac_inliers'][0][0])
        rat = float(data['inlier_ratio'][0][0])
        err = float(data['mean_reprojection_error'][0][0])
        
        all_raw.append(raw)
        all_inl.append(inl)
        all_rat.append(rat)
        all_err.append(err)
        
        print(f"{frame:>5} | {raw:>11} | {inl:>7} | {rat:>9.1f} | {err:>9.2f}")
    else:
        print(f"{frame:>5} | (no stats available)")

if all_raw:
    print("-" * 55)
    print(f"MEAN  | {np.mean(all_raw):>11.0f} | {np.mean(all_inl):>7.0f} | {np.mean(all_rat):>9.1f} | {np.mean(all_err):>9.2f}")
    print(f"MIN   | {np.min(all_raw):>11.0f} | {np.min(all_inl):>7.0f} | {np.min(all_rat):>9.1f} | {np.min(all_err):>9.2f}")
    print(f"MAX   | {np.max(all_raw):>11.0f} | {np.max(all_inl):>7.0f} | {np.max(all_rat):>9.1f} | {np.max(all_err):>9.2f}")
