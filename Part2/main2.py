import sys
import os
import cv2
import numpy as np
from scipy.io import loadmat, savemat
from scipy.spatial.distance import cdist  

def detect_and_match_features(img1, img2):
    # Exemplo com SIFT (implemente matching)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Filtrar bons matches (ex.: ratio test)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    return pts1, pts2

def extract_3d_points(pts1, pts2, depth1, depth2, K):
    # Converta pontos 2D para 3D usando depth e K
    # K é (3,3), depth é (H,W)
    # Para cada ponto (u,v), Z = depth[v,u], X = (u - cx)/fx * Z, Y = (v - cy)/fy * Z
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    pts3d1 = []
    pts3d2 = []
    for (u1, v1), (u2, v2) in zip(pts1, pts2):
        z1 = depth1[int(v1), int(u1)]
        z2 = depth2[int(v2), int(u2)]
        if z1 > 0 and z2 > 0:  # Verifique profundidade válida
            x1 = (u1 - cx) * z1 / fx
            y1 = (v1 - cy) * z1 / fy
            pts3d1.append([x1, y1, z1])
            x2 = (u2 - cx) * z2 / fx
            y2 = (v2 - cy) * z2 / fy
            pts3d2.append([x2, y2, z2])
    return np.array(pts3d1), np.array(pts3d2)

def procrustes(pts_A, pts_B):
    # Implemente Procrustes: centralize, SVD para R, compute T
    centroid_A = np.mean(pts_A, axis=0)
    centroid_B = np.mean(pts_B, axis=0)
    A_centered = pts_A - centroid_A
    B_centered = pts_B - centroid_B
    H = B_centered.T @ A_centered
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # Correção para reflexão
        Vt[-1, :] *= -1
        R = U @ Vt
    T = centroid_B - R @ centroid_A
    return R, T

def ransac_procrustes(pts_A, pts_B, threshold=0.01, max_iter=1000, min_samples=3):
    # RANSAC para Procrustes
    best_R, best_T = None, None
    best_inliers = 0
    n_points = len(pts_A)
    for _ in range(max_iter):
        indices = np.random.choice(n_points, min_samples, replace=False)
        R, T = procrustes(pts_A[indices], pts_B[indices])
        transformed = (R @ pts_A.T).T + T
        distances = np.linalg.norm(transformed - pts_B, axis=1)
        inliers = np.sum(distances < threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            best_R, best_T = R, T
    return best_R, best_T

def main():
    if len(sys.argv) != 4:
        print("Uso: python main2.py path_to_refimgdir path_images_dir path_output_dir")
        return
    
    ref_dir = sys.argv[1]
    images_dir = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar referência
    ref_rgb = cv2.imread(os.path.join(ref_dir, 'templatergb.jpg'))
    ref_data = loadmat(os.path.join(ref_dir, 'templatedepth.mat'))
    ref_depth = ref_data['depth']
    K = ref_data['K']
    
    # Listar arquivos de sequência 
    rgb_files = sorted([f for f in os.listdir(images_dir) if f.startswith('frame_') and f.endswith('.jpg')])
    depth_files = sorted([f for f in os.listdir(images_dir) if f.startswith('frame_') and f.endswith('.mat')])
    
    for rgb_file, depth_file in zip(rgb_files, depth_files):
        frame_num = rgb_file.split('_')[1].split('.')[0]  
        curr_rgb = cv2.imread(os.path.join(images_dir, rgb_file))
        curr_data = loadmat(os.path.join(images_dir, depth_file))
        curr_depth = curr_data['depth']
        
        # Detectar e matching
        pts_ref, pts_curr = detect_and_match_features(ref_rgb, curr_rgb)
        
        # Extrair 3D
        pts3d_ref, pts3d_curr = extract_3d_points(pts_ref, pts_curr, ref_depth, curr_depth, K)
        
        if len(pts3d_ref) < 3:
            continue  # Poucos pontos
        
        # RANSAC + Procrustes
        R, T = ransac_procrustes(pts3d_ref, pts3d_curr)
        
        # Salvar
        savemat(os.path.join(output_dir, f'transform_{frame_num}.mat'), {'R': R, 'T': T})

if __name__ == "__main__":
    main()