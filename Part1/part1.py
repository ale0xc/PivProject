import numpy as np
import scipy.io as sio
import scipy.spatial.distance as dist
import os

def normalize_points(points):
    """ Normaliza pontos para melhorar a estabilidade do DLT """
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    avg_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    scale = np.sqrt(2) / avg_dist
    
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    # Transforma pontos para coordenadas homogéneas e aplica T
    points_h = np.column_stack((points, np.ones(len(points))))
    points_norm = (T @ points_h.T).T
    return points_norm[:, :2], T

def compute_homography_dlt(src, dst):
    """ Calcula H usando DLT (Direct Linear Transform) [cite: 52] """
    if len(src) < 4: return None
    
    # Normalização (Crucial para precisão)
    src_norm, T_src = normalize_points(src)
    dst_norm, T_dst = normalize_points(dst)
    
    num_points = len(src)
    A = []
    
    for i in range(num_points):
        x, y = src_norm[i][0], src_norm[i][1]
        u, v = dst_norm[i][0], dst_norm[i][1]
        
        # Cria as duas linhas da matriz A para cada ponto
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        
    A = np.array(A)
    
    # SVD para resolver Ah = 0. A solução é o último vetor de V
    U, S, Vh = np.linalg.svd(A)
    H_norm = Vh[-1, :].reshape(3, 3)
    
    # Desnormalizar: H = inv(T_dst) * H_norm * T_src
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    return H / H[2, 2] # Normalizar para que o ultimo elemento seja 1

def part1(path_ref, path_imgs, path_feats, path_out):
    # 1. Carregar features do template
    template_data = sio.loadmat(os.path.join(path_feats, "template_features.mat"))
    kp_ref = template_data["keypoints"]
    desc_ref = template_data["descriptors"].astype(np.float32)

    # Listar ficheiros de features das imagens de input
    feat_files = sorted([f for f in os.listdir(path_feats) if f != "template_features.mat"])

    for f in feat_files:
        # Carregar features do frame atual
        frame_data = sio.loadmat(os.path.join(path_feats, f))
        kp_frame = frame_data["keypoints"]
        desc_frame = frame_data["descriptors"].astype(np.float32)

        if len(kp_frame) < 4:
            continue

        # --- MATCHING (Sem CV2) ---
        # Usamos distancia Euclidiana. Para SIFT funciona bem.
        # cdist calcula a distancia entre todos os pares
        dists = dist.cdist(desc_frame, desc_ref, metric='euclidean')
        
        # Lowe's Ratio Test (Simples)
        # Encontra os 2 vizinhos mais próximos para cada ponto do frame
        sorted_indices = np.argsort(dists, axis=1)
        matches_src = []
        matches_dst = []
        
        ratio = 0.75
        for i in range(len(dists)):
            idx1, idx2 = sorted_indices[i, 0], sorted_indices[i, 1]
            if dists[i, idx1] < ratio * dists[i, idx2]:
                matches_src.append(kp_frame[i])
                matches_dst.append(kp_ref[idx1])
        
        matches_src = np.array(matches_src)
        matches_dst = np.array(matches_dst)

        if len(matches_src) < 4:
            print(f"Matches insuficientes em {f}")
            continue

        # --- RANSAC --- 
        best_H = np.eye(3)
        max_inliers = 0
        threshold = 3.0 # Distância em pixels para considerar inlier
        iterations = 5000 # Ajustar conforme necessário

        for _ in range(iterations):
            # 1. Amostrar 4 pontos aleatórios
            idx = np.random.choice(len(matches_src), 4, replace=False)
            src_sample = matches_src[idx]
            dst_sample = matches_dst[idx]

            # 2. Calcular Homografia candidata
            H_curr = compute_homography_dlt(src_sample, dst_sample)
            if H_curr is None: continue

            # 3. Contar inliers
            # Projetar todos os pontos src usando H_curr
            ones = np.ones((len(matches_src), 1))
            src_h = np.hstack((matches_src, ones))
            projected = (H_curr @ src_h.T).T
            
            # Evitar divisão por zero
            with np.errstate(divide='ignore', invalid='ignore'):
                projected = projected[:, :2] / projected[:, 2:]
            
            # Calcular erro (distancia para os pontos de destino reais)
            errors = np.linalg.norm(projected - matches_dst, axis=1)
            inliers_mask = errors < threshold
            num_inliers = np.sum(inliers_mask)

            # 4. Atualizar melhor modelo
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_H = H_curr
                best_inliers_mask = inliers_mask

        # --- REFINAMENTO --- 
        # Recalcular H usando TODOS os inliers do melhor modelo (Mínimos Quadrados implícito no DLT com N pontos)
        if max_inliers > 4:
            final_src = matches_src[best_inliers_mask]
            final_dst = matches_dst[best_inliers_mask]
            best_H = compute_homography_dlt(final_src, final_dst)

        # --- GUARDAR RESULTADO --- [cite: 95]
        out_name = "homography_" + f.replace(".mat", ".mat").split('_')[-1] # formata homography_NNNN.mat
        sio.savemat(os.path.join(path_out, out_name), {"H": best_H})
        print(f"Processado {f}: {max_inliers} inliers. H guardado.")