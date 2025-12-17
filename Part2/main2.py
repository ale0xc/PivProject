import argparse
import sys
import os
import glob
import numpy as np
import cv2
import scipy.io as sio

def load_mat_depth_k(filepath):
    try:
        data = sio.loadmat(filepath)
        depth = data['depth']
        K = data['K']
        return depth, K
    except Exception as e:
        print(f"Error during loading of {filepath}: {e}")
        return None, None

def get_3d_points(pixel_coords, depth_map, K):
    """
    Compute 3D point cloud based on 2D coordenates
    """
    # Extraction des intrinsèques
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

def estimate_rigid_transform(pts_curr, pts_ref):
    """
    Estimate (R,T) thanks to Procruste method
    """
    if pts_curr.shape[0] < 3:
        return None, None
    
    mean_curr = np.mean(pts_curr, axis=0)
    mean_ref = np.mean(pts_ref, axis=0)

    # Centering points
    curr_centered = pts_curr - mean_curr
    ref_centered = pts_ref - mean_ref

    # Computing H
    H = np.dot(curr_centered.T, ref_centered)

    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # 5. Computing R
    R = np.dot(Vt.T, U.T)

    # 6. If det < 0
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Computing T
    T = mean_ref.reshape(3, 1) - np.dot(R, mean_curr.reshape(3, 1))

    return R, T

def process_alignment(ref_img, ref_depth, ref_K, curr_img, curr_depth, curr_K):
    """
    Détecte les features, trouve les correspondances, lève en 3D et calcule R, T.
    """
    # SIFT
    sift = cv2.SIFT_create()
    
    kp_ref, des_ref = sift.detectAndCompute(ref_img, None)
    kp_curr, des_curr = sift.detectAndCompute(curr_img, None)

    # Matching 
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_curr, des_ref, k=2) 
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < 4:
        print("Not enough pairs found")
        return None, None

    pts_uv_curr = np.float32([kp_curr[m.queryIdx].pt for m in good_matches])
    pts_uv_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches])

    # 3D conversion
    pts_3d_curr = get_3d_points(pts_uv_curr, curr_depth, curr_K)
    pts_3d_ref = get_3d_points(pts_uv_ref, ref_depth, ref_K)
    
    valid_mask = ~np.isnan(pts_3d_curr).any(axis=1) & ~np.isnan(pts_3d_ref).any(axis=1)
    pts_3d_curr_clean = pts_3d_curr[valid_mask]
    pts_3d_ref_clean = pts_3d_ref[valid_mask]
    
    if len(pts_3d_curr_clean) < 4:
        print("Not enough 3D valid points")
        return None, None

    R, T, inliers = ransac_pose_estimation(pts_3d_curr_clean, pts_3d_ref_clean, num_iterations=2000, threshold=0.05)
    
    if R is not None:
        ratio = np.sum(inliers) / len(pts_3d_curr_clean)
        print(f"  RANSAC: {np.sum(inliers)} inliers sur {len(pts_3d_curr_clean)} points ({ratio:.2%})")
        
    return R, T
    

def ransac_pose_estimation(src_points, dst_points, num_iterations=1000, threshold=0.02):
    """
    Estimation with RANSAC
    """
    if len(src_points) < 4:
        return None, None, None

    best_inliers_count = -1
    best_inliers_mask = None
    N = src_points.shape[0]
    
    for i in range(num_iterations):
        sample_indices = np.random.choice(N, size=4, replace=False) 
        src_sample = src_points[sample_indices]
        dst_sample = dst_points[sample_indices]
        
        # 2. Estimer le modèle sur cet échantillon
        R_sample, T_sample = estimate_rigid_transform(src_sample, dst_sample)
        
        if R_sample is None:
            continue
            
        src_transformed = (np.dot(R_sample, src_points.T) + T_sample).T
        
        diff = src_transformed - dst_points
        errors = np.linalg.norm(diff, axis=1) 
        
        current_inliers_mask = errors < threshold
        current_inliers_count = np.sum(current_inliers_mask)
        
        if current_inliers_count > best_inliers_count:
            best_inliers_count = current_inliers_count
            best_inliers_mask = current_inliers_mask

    if best_inliers_mask is not None and best_inliers_count >= 4:
        final_src = src_points[best_inliers_mask]
        final_dst = dst_points[best_inliers_mask]
        best_R, best_T = estimate_rigid_transform(final_src, final_dst)
        return best_R, best_T, best_inliers_mask
    else:
        print("RANSAC didnt find anything")
        return None, None, None
    
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pipeline to incorporate depth images.")
    parser.add_argument('path_to_refimgdir', type=str, help="The path to the image/depth DIRECTORY with the template data:  templatergb.jpg, templatedepth.mat and cam.mat ")
    parser.add_argument('path_images_dir', type=str, help="Path to the directory with the image and depth sequence")
    parser.add_argument('path_output_dir', type=str, help="Path to the output results")
    args = parser.parse_args()

    # Creation output folder
    if not os.path.exists(args.path_output_dir):
        os.makedirs(args.path_output_dir)

    # 1. Load Template data 
    ref_rgb_path = os.path.join(args.path_to_refimgdir, 'templatergb.jpg')
    ref_depth_path = os.path.join(args.path_to_refimgdir, 'templatedepth.mat')

    ref_img = cv2.imread(ref_rgb_path)
    if ref_img is None:
        print("Error: Can't Load Template image.")
        sys.exit(1)
        
    try:
        ref_depth_data = sio.loadmat(ref_depth_path)
        ref_depth = ref_depth_data['depth'] 
        ref_K = ref_depth_data['K']
    except:
        print("Errorr loading reference depth mat")
        sys.exit(1)

    print("Loading template successful")

    
    image_files = sorted(glob.glob(os.path.join(args.path_images_dir, '*.jpg')))
    
    for img_path in image_files:
        basename = os.path.basename(img_path)
        name_root, _ = os.path.splitext(basename)
        depth_path = os.path.join(args.path_images_dir, name_root + '.mat')
        
        if not os.path.exists(depth_path):
            print(f"Attention: Fichier depth manquant pour {basename}, ignoré.")
            continue

        # Chargement image courante
        curr_img = cv2.imread(img_path)
        
        # Chargement depth et K courants
        curr_depth, curr_K = load_mat_depth_k(depth_path)
        if curr_depth is None or curr_K is None:
            continue

        print(f"Traitement de {basename}...")

        R, T = process_alignment(ref_img, ref_depth, ref_K, curr_img, curr_depth, curr_K)

        if R is not None:
            try:
                seq_num = name_root.split('_')[-1]
                out_name = f"transform_{seq_num}.mat"
            except:
                out_name = f"transform_{name_root}.mat"  
            sio.savemat(os.path.join(args.path_output_dir, out_name), {"R": R, "T": T}) 

        warped_result = warp_current_to_template(ref_depth, ref_K, curr_img, curr_K, R, T)

        h_vis, w_vis = ref_img.shape[:2]
        curr_resized = cv2.resize(curr_img, (w_vis, h_vis))
         
        vis = np.zeros((h_vis, w_vis*3, 3), dtype=np.uint8)
        vis[:, :w_vis] = curr_resized
        vis[:, w_vis:w_vis*2] = warped_result
        vis[:, w_vis*2:] = ref_img
            
        # Annotations
        cv2.putText(vis, "Camera (Input)", (20,50), 0, 1, (0,0,255), 2)
        cv2.putText(vis, "Stabilised (Result)", (w_vis+20,50), 0, 1, (0,255,0), 2)
        cv2.putText(vis, "Template (Goal)", (w_vis*2+20,50), 0, 1, (255,0,0), 2)

        cv2.imwrite(os.path.join(args.path_output_dir, f"vis_{name_root}.jpg"), vis)
        print(f"Succès -> T: {T.T}") 