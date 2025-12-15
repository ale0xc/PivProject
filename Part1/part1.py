import numpy as np
from PIL import Image
import os 
import scipy.io as sio
import cv2              # For FLANN only 

def get_data_from_mat(path):
    """
    Load file .mat : :
    Matrix (130, N) where :
    - Lines 0-1 : Coordenates (x, y)
    - Lines 2-130 : Descriptors (128 dimensions)
    (see part1.py)
    """
    try: 
        data = sio.loadmat(path)
        combined = data['kp'].T
        return combined[:, 0:2], combined[:, 2:]  #kp, des
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None 

def match_features(des_ref, des_curr, ratio=0.75):
    """
    Match features using FLANN based matcher
    """
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    # flann = cv2.FlannBasedMatcher(index_params, search_params)

    # matches = flann.knnMatch(des_curr.astype(np.float32), des_ref.astype(np.float32), k=2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_curr,des_ref,k=2)
    matches_curr_idx = [] 
    matches_ref_idx = [] 
    
    # 4. Filter (Lowe's Ratio Test)
    for m, n in matches:
        if m.distance < ratio * n.distance:
            matches_ref_idx.append(m.trainIdx)
            matches_curr_idx.append(m.queryIdx)
            
    return np.array(matches_curr_idx), np.array(matches_ref_idx)

def normalize_points(pts):
    
    mean = np.mean(pts, axis=0)
    dists = np.linalg.norm(pts-mean, axis=1)
    avg_dist = np.mean(dists)
    scale = np.sqrt(2) / avg_dist

    T = np.array([
        [scale, 0,     -scale * mean[0]],
        [0,     scale, -scale * mean[1]],
        [0,     0,     1               ]
    ])
    
    pts_h = np.column_stack((pts, np.ones(pts.shape[0])))
    pts_norm = (T @ pts_h.T).T
    
    return pts_norm[:, :2], T

def ransac_homography(p1_norm, p2_norm, N, num_iter=5000, threshold_norm=0.05):

    p1_h_norm = np.column_stack((p1_norm, np.ones(N)))

    best_H_norm = None
    best_inliers_count = -1
    best_mask = np.zeros(N, dtype=bool)

    for _ in range(num_iter):
        # 1 Randomly select 4 points
        idx = np.random.choice(N, 4, replace=False)
        s1 = p1_norm[idx]
        s2 = p2_norm[idx]

        # 2 Build matrix A
        A_list = []
        for k in range(4):
            x, y = s1[k]
            u, v = s2[k]
            A_list.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A_list.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

        A = np.array(A_list)

        # 3 Solve Ah = 0 using SVD
        U, S, Vt = np.linalg.svd(A)
        H_cand = Vt[-1].reshape(3, 3)

        # 4 Compute projections and errors (P2_proj = H * P1)
        p2_proj_h = (H_cand @ p1_h_norm.T).T
        w = p2_proj_h[:, 2:3]
        w[np.abs(w) < 1e-10] = 1e-10 
        p2_proj_xy = p2_proj_h[:, :2] / w

        errors = np.linalg.norm(p2_norm - p2_proj_xy, axis=1)

        # 5 Count inliers
        current_inliers = errors < threshold_norm
        count = np.sum(current_inliers)
        
        if count > best_inliers_count:
            best_inliers_count = count
            best_H_norm = H_cand
            best_mask = current_inliers
            
        return best_inliers_count, best_H_norm, best_mask

def compute_homography(pts1, pts2, num_iter=2000, threshold_norm=0.05):
    N = len(pts1)
    if N < 4: 
        return np.eye(3)

    # 1. Normalization
    p1_norm, T1 = normalize_points(pts1)
    p2_norm, T2 = normalize_points(pts2)

    # 2. RANSAC Homography Estimation
    best_inliers_count, H_norm, best_mask = ransac_homography(p1_norm, p2_norm, N, num_iter, threshold_norm)

    # 3. Compute final Homography using all inliers
    if best_inliers_count >= 4:
        p1_final = p1_norm[best_mask]
        p2_final = p2_norm[best_mask]
        N_inliers = len(p1_final)
        
        # 3.1 Build matrix A with all inliers
        A_final = []
        for k in range(N_inliers):
            x, y = p1_final[k]
            u, v = p2_final[k]
            A_final.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A_final.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
            
        _, _, Vt_final = np.linalg.svd(np.array(A_final))
        H_norm_final = Vt_final[-1].reshape(3, 3)
        
        # 3.2 Denormalize 
        H_final = np.linalg.inv(T2) @ H_norm_final @ T1
        
        # 3.3 Normalize so that H[2,2] = 1
        if H_final[2, 2] != 0:
            H_final = H_final / H_final[2, 2]  
        return H_final
    
    else:
        print("Warning: Not enough inliers found. Returning identity matrix.")
        return np.eye(3)


def part1(path1, path2, path3, path4):
    
    # 1. Load template
    ref_name = os.path.splitext(os.path.basename(path1))[0]
    ref_mat_path = os.path.join(path3, ref_name + ".mat")
    kp_ref, des_ref = get_data_from_mat(ref_mat_path)

    # 2. Iterate on all images in folder
    for fname in os.listdir(path3):
        if fname == ref_name + ".mat":
            continue
        if not fname.lower().endswith('.mat'):
            continue
        curr_mat_path = os.path.join(path3, fname)

        ## 2.1 Compute keypoints and descriptors
        kp_curr, des_curr = get_data_from_mat(curr_mat_path)

        ## 2.2 Match features
        idx_curr, idx_ref = match_features(des_ref, des_curr, ratio = 0.60)

        ## 2.3 Compute Homography using RANSAC
        if len(idx_curr) < 4:
            H = np.eye(3)
        else:
            H = compute_homography(kp_curr[idx_curr], kp_ref[idx_ref], num_iter=2000, threshold_norm=0.05)
            
        # Sauvegarde
        base_name = os.path.splitext(fname)[0]
        seq_num = base_name.split('_')[-1]
        out_path = os.path.join(path4, f"homography_{seq_num}.mat")
        sio.savemat(out_path, {"H": H})


