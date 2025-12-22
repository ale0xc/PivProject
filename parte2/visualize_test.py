import sys
import os
import cv2
import numpy as np
from scipy.io import loadmat

def detect_and_match_features(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    return pts1, pts2

def generate_point_cloud(depth, rgb, K):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    points = []
    colors = []
    h, w = depth.shape
    for v in range(h):
        for u in range(w):
            z = depth[v, u] #/ 1000.0
            if z > 0:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                color = rgb[v, u]
                colors.append(color)
    return np.array(points), np.array(colors)

def extract_3d_points(pts1, pts2, depth1, depth2, K1, K2):
    fx1, fy1, cx1, cy1 = K1[0,0], K1[1,1], K1[0,2], K1[1,2]
    fx2, fy2, cx2, cy2 = K2[0,0], K2[1,1], K2[0,2], K2[1,2]
    pts3d1 = []
    pts3d2 = []
    for (u1, v1), (u2, v2) in zip(pts1, pts2):
        z1 = depth1[int(v1), int(u1)] #/ 1000.0
        z2 = depth2[int(v2), int(u2)] #/ 1000.0
        if z1 > 0 and z2 > 0:
            x1 = (u1 - cx1) * z1 / fx1
            y1 = (v1 - cy1) * z1 / fy1
            pts3d1.append([x1, y1, z1])
            x2 = (u2 - cx2) * z2 / fx2
            y2 = (v2 - cy2) * z2 / fy2
            pts3d2.append([x2, y2, z2])
    return np.array(pts3d1), np.array(pts3d2)

def main():
    if len(sys.argv) != 4:
        print("Uso: python visulaize_teste.py path_to_refimgdir path_images_dir path_output_dir")
        return
    
    ref_dir = sys.argv[1]
    images_dir = sys.argv[2]
    output_dir = sys.argv[3]

    # Carregar referência
    ref_rgb = cv2.imread(os.path.join(ref_dir, 'templatergb.jpg'))
    ref_data = loadmat(os.path.join(ref_dir, 'templatedepth.mat'))
    ref_depth = ref_data['depth']
    K_ref = ref_data['K']

    # Gerar nuvem de pontos completa da referência
    ref_points, ref_colors = generate_point_cloud(ref_depth, ref_rgb, K_ref)
    print(f"Nuvem de referência: {len(ref_points)} pontos")

    # Listar arquivos de transformação
    transform_files = [f for f in os.listdir(output_dir) if f.startswith('transform_') and f.endswith('.mat')]
    if not transform_files:
        print("Nenhum arquivo de transformação encontrado.")
        return

    all_transformed_points = []
    all_transformed_colors = []

    for transform_file in transform_files:
        frame_num = transform_file.split('_')[1].split('.')[0]

        # Carregar frame
        rgb_file = f'frame_{frame_num}.jpg'
        depth_file = f'frame_{frame_num}.mat'
        curr_rgb = cv2.imread(os.path.join(images_dir, rgb_file))
        curr_data = loadmat(os.path.join(images_dir, depth_file))
        curr_depth = curr_data['depth']
        curr_K = curr_data['K']

        # Detectar e matching (para cálculo de distâncias)
        pts_ref, pts_curr = detect_and_match_features(ref_rgb, curr_rgb)

        # Extrair 3D (para cálculo de distâncias)
        pts3d_ref, pts3d_curr = extract_3d_points(pts_ref, pts_curr, ref_depth, curr_depth, K_ref, curr_K)

        print(f"Frame {frame_num}: Pontos 3D extraídos: {len(pts3d_ref)}")

        # Carregar transformação
        transform_path = os.path.join(output_dir, transform_file)
        transform_data = loadmat(transform_path)
        R = transform_data['R']
        T = transform_data['T']

        # Aplicar transformação aos pontos do frame atual (para distâncias)
        transformed_pts = (R @ pts3d_curr.T).T + T

        # Calcular distâncias
        distances = np.linalg.norm(transformed_pts - pts3d_ref, axis=1)
        mean_dist = np.mean(distances)
        max_dist = np.max(distances)
        median_dist = np.median(distances)

        print(f"Frame {frame_num}: Distância média: {mean_dist:.4f} m, Máxima: {max_dist:.4f} m, Mediana: {median_dist:.4f} m")

        # Gerar nuvem completa do frame
        curr_points, curr_colors = generate_point_cloud(curr_depth, curr_rgb, curr_K)
        print(f"Frame {frame_num}: Nuvem completa: {len(curr_points)} pontos")

        # Aplicar transformação à nuvem completa
        transformed_full_points = (R @ curr_points.T).T + T

        # Adicionar à lista total
        all_transformed_points.extend(transformed_full_points)
        all_transformed_colors.extend(curr_colors)

    # Salvar nuvem combinada total (referência + todas transformadas) para visualização
    ply_file = os.path.join(os.getcwd(), 'pointcloud_all.ply')
    with open(ply_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(ref_points) + len(all_transformed_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        # Pontos de referência
        for pt, color in zip(ref_points, ref_colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {color[2]} {color[1]} {color[0]}\n")  # BGR to RGB
        # Todos os pontos transformados
        for pt, color in zip(all_transformed_points, all_transformed_colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {color[2]} {color[1]} {color[0]}\n")  # BGR to RGB
    print(f"Nuvem combinada total salva em: {ply_file}")

if __name__ == "__main__":
    main()
