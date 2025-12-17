import sys
import os
import glob
import numpy as np
import cv2
import scipy.io as sio

# =========================================================
# PARTIE 1 : OUTILS DE CHARGEMENT ET GÉOMÉTRIE
# =========================================================

def load_mat_depth_k(filepath):
    """ Charge le .mat contenant {"depth": depth, "K": K} """
    try:
        data = sio.loadmat(filepath)
        depth = data['depth']
        K = data['K']
        
        # VÉRIFICATION D'ÉCHELLE (Mètres vs Millimètres)
        # Si la profondeur max est > 100, c'est probablement des mm -> conversion en m
        if np.nanmax(depth) > 100:
            depth = depth / 1000.0
            
        return depth, K
    except Exception as e:
        print(f"Erreur chargement {os.path.basename(filepath)}: {e}")
        return None, None

def get_3d_points(pixel_coords, depth_map, K):
    """ Back-projection: 2D (pixels) -> 3D (caméra) """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    points_3d = []
    
    for uv in pixel_coords:
        u, v = int(round(uv[0])), int(round(uv[1]))
        
        # Vérif limites image
        if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
            z = depth_map[v, u]
            # On ignore les profondeurs nulles ou NaN
            if z > 0 and not np.isnan(z):
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points_3d.append([x, y, z])
            else:
                points_3d.append([np.nan, np.nan, np.nan])
        else:
            points_3d.append([np.nan, np.nan, np.nan])
            
    return np.array(points_3d)

# =========================================================
# PARTIE 2 : MATHÉMATIQUES (PROCRUSTES & RANSAC)
# =========================================================

def estimate_rigid_transform(points_src, points_dst):
    """ Procrustes Orthogonal (SVD) """
    if points_src.shape[0] < 3:
        return None, None

    # Centrage
    centroid_src = np.mean(points_src, axis=0)
    centroid_dst = np.mean(points_dst, axis=0)
    src_centered = points_src - centroid_src
    dst_centered = points_dst - centroid_dst

    # Covariance H
    H = np.dot(src_centered.T, dst_centered)

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Correction réflexion (det must be 1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Translation
    T = centroid_dst.reshape(3, 1) - np.dot(R, centroid_src.reshape(3, 1))

    return R, T

def ransac_pose_estimation(src_points, dst_points, num_iterations=1000, threshold=0.03):
    """ RANSAC pour éliminer les outliers (main, clavier, bruit) """
    if len(src_points) < 4:
        return None, None, None

    best_inliers_count = -1
    best_inliers_mask = None
    N = src_points.shape[0]
    

    for i in range(num_iterations):
        sample_indices = np.random.choice(N, size=4, replace=False)
        src_sample = src_points[sample_indices]
        dst_sample = dst_points[sample_indices]
        
        R_sample, T_sample = estimate_rigid_transform(src_sample, dst_sample)
        if R_sample is None: continue
            
        src_transformed = (np.dot(R_sample, src_points.T) + T_sample).T
        
        # Calcul erreur
        diff = src_transformed - dst_points
        errors = np.linalg.norm(diff, axis=1)
        
        # Compter inliers
        current_inliers_mask = errors < threshold
        current_inliers_count = np.sum(current_inliers_mask)
        
        if current_inliers_count > best_inliers_count:
            best_inliers_count = current_inliers_count
            best_inliers_mask = current_inliers_mask

    # Raffinement final avec tous les inliers
    if best_inliers_mask is not None and best_inliers_count >= 4:
        final_src = src_points[best_inliers_mask]
        final_dst = dst_points[best_inliers_mask]
        best_R, best_T = estimate_rigid_transform(final_src, final_dst)
        return best_R, best_T, best_inliers_mask
    
    return None, None, None

# =========================================================
# PARTIE 3 : VISUALISATION (WARPING)
# =========================================================

def warp_current_to_template(ref_depth, ref_K, curr_img, curr_K, R, T):
    """ Déforme l'image courante pour qu'elle s'aligne sur la référence """
    h, w = ref_depth.shape
    
    # 1. Grille Pixels Reference
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    
    # 2. Back-projection Reference (Pixels -> 3D Ref)
    cx_ref, cy_ref = ref_K[0, 2], ref_K[1, 2]
    fx_ref, fy_ref = ref_K[0, 0], ref_K[1, 1]
    z_ref = ref_depth
    x_ref = (u_grid - cx_ref) * z_ref / fx_ref
    y_ref = (v_grid - cy_ref) * z_ref / fy_ref
    
    # Points 3D Ref applatis
    P_ref_flat = np.stack((x_ref, y_ref, z_ref), axis=-1).reshape(-1, 3)

    # 3. Transfo Inverse : Ref -> Current
    # P_curr = R.T * (P_ref - T)
    P_curr_flat = np.dot(P_ref_flat - T.flatten(), R) # Note: R ici agit comme R.T dans la formule math standard car numpy dot

    # 4. Projection : 3D Current -> Pixels Current
    cx_curr, cy_curr = curr_K[0, 2], curr_K[1, 2]
    fx_curr, fy_curr = curr_K[0, 0], curr_K[1, 1]
    
    x_c, y_c, z_c = P_curr_flat[:, 0], P_curr_flat[:, 1], P_curr_flat[:, 2]
    
    # Sécurité Z
    z_c_safe = np.where(z_c > 0.001, z_c, 0.001)
    
    u_map = (x_c * fx_curr / z_c_safe) + cx_curr
    v_map = (y_c * fy_curr / z_c_safe) + cy_curr
    
    # 5. Remap
    map_x = u_map.reshape(h, w).astype(np.float32)
    map_y = v_map.reshape(h, w).astype(np.float32)
    
    warped = cv2.remap(curr_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Masquer les zones sans profondeur dans la ref
    mask_invalid = (z_ref <= 0) | np.isnan(z_ref)
    warped[mask_invalid] = 0
    
    return warped

# =========================================================
# PARTIE 4 : MAIN LOOP & PROCESS
# =========================================================

def process_alignment(ref_img, ref_depth, ref_K, curr_img, curr_depth, curr_K):
    # 1. SIFT
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_img, None)
    kp_curr, des_curr = sift.detectAndCompute(curr_img, None)
    
    if des_curr is None or des_ref is None: return None, None

    # 2. Matching (Lowe's Ratio Test)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_curr, des_ref, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            
    if len(good) < 5: return None, None

    # 3. Lifting 3D
    pts_uv_curr = np.float32([kp_curr[m.queryIdx].pt for m in good])
    pts_uv_ref = np.float32([kp_ref[m.trainIdx].pt for m in good])
    
    pts_3d_curr = get_3d_points(pts_uv_curr, curr_depth, curr_K)
    pts_3d_ref = get_3d_points(pts_uv_ref, ref_depth, ref_K)
    
    # Nettoyage NaNs
    valid = ~np.isnan(pts_3d_curr).any(axis=1) & ~np.isnan(pts_3d_ref).any(axis=1)
    p_curr_clean = pts_3d_curr[valid]
    p_ref_clean = pts_3d_ref[valid]
    
    if len(p_curr_clean) < 4: return None, None

    # 4. RANSAC Estimation
    # Threshold 0.03 = 3cm de tolérance
    R, T, inliers = ransac_pose_estimation(p_curr_clean, p_ref_clean, num_iterations=500, threshold=0.03)
    
    if R is not None:
        print(f"  > RANSAC: {np.sum(inliers)} inliers / {len(p_curr_clean)} matches")
        
    return R, T

def main():
    if len(sys.argv) != 4:
        print("Usage: python main2.py <ref_dir> <images_dir> <output_dir>")
        sys.exit(1)

    path_ref = sys.argv[1]
    path_imgs = sys.argv[2]
    path_out = sys.argv[3]
    
    if not os.path.exists(path_out): os.makedirs(path_out)

    # Chargement Reference
    print("Chargement Reference...")
    ref_rgb = cv2.imread(os.path.join(path_ref, 'templatergb.jpg'))
    
    # Chargement flexible depth/K ref
    try:
        ref_d_data = sio.loadmat(os.path.join(path_ref, 'templatedepth.mat'))
        ref_depth = ref_d_data['depth']
        if 'K' in ref_d_data: 
            ref_K = ref_d_data['K']
        else:
            ref_cam = sio.loadmat(os.path.join(path_ref, 'cam.mat'))
            ref_K = ref_cam['K']
            
        # Check scale ref
        if np.nanmax(ref_depth) > 100: ref_depth /= 1000.0
            
    except Exception as e:
        print(f"Erreur chargement ref data: {e}")
        sys.exit(1)

    # Boucle Images
    img_files = sorted(glob.glob(os.path.join(path_imgs, '*.jpg')))
    print(f"Trouvé {len(img_files)} images dans la séquence.")

    for img_p in img_files:
        basename = os.path.basename(img_p)
        name_root = os.path.splitext(basename)[0]
        
        # Trouver depth associée
        depth_p = os.path.join(path_imgs, name_root + '.mat')
        if not os.path.exists(depth_p): continue
            
        # Chargement
        curr_img = cv2.imread(img_p)
        curr_depth, curr_K = load_mat_depth_k(depth_p)
        
        if curr_depth is None: continue

        print(f"--- Traitement {basename} ---")
        
        # Calcul
        R, T = process_alignment(ref_rgb, ref_depth, ref_K, curr_img, curr_depth, curr_K)
        
        if R is not None:
            # 1. Save .mat
            # Extraire le numéro NNNN
            try:
                seq_num = name_root.split('_')[-1]
                out_name = f"transform_{seq_num}.mat"
            except:
                out_name = f"transform_{name_root}.mat"
                
            sio.savemat(os.path.join(path_out, out_name), {"R": R, "T": T})
            
            # 2. Visualization (La Preuve !)
            warped = warp_current_to_template(ref_depth, ref_K, curr_img, curr_K, R, T)
            
            # Assemblage : [Input Redimensionné] | [Resultat Warpé] | [Template Original]
            h_vis, w_vis = ref_rgb.shape[:2]
            curr_resized = cv2.resize(curr_img, (w_vis, h_vis))
            
            vis = np.zeros((h_vis, w_vis*3, 3), dtype=np.uint8)
            vis[:, :w_vis] = curr_resized
            vis[:, w_vis:w_vis*2] = warped
            vis[:, w_vis*2:] = ref_rgb
            
            # Annotations
            cv2.putText(vis, "Camera (Input)", (20,50), 0, 1, (0,0,255), 2)
            cv2.putText(vis, "Stabilised (Result)", (w_vis+20,50), 0, 1, (0,255,0), 2)
            cv2.putText(vis, "Template (Goal)", (w_vis*2+20,50), 0, 1, (255,0,0), 2)

            cv2.imwrite(os.path.join(path_out, f"vis_{name_root}.jpg"), vis)
            print(f"Succès -> T: {T.T}")


if __name__ == "__main__":
    main()