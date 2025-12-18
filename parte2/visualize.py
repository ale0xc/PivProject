import sys
import os
import numpy as np
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def depth_to_point_cloud(depth, K, rgb=None):
    # Converte mapa de profundidade em nuvem de pontos 3D
    h, w = depth.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    valid = z > 0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x[valid], y[valid], z[valid]], axis=1)
    if rgb is not None:
        colors = rgb[valid] / 255.0  # Normalizar para [0,1]
        colors = colors[:, [2, 1, 0]]
        return points, colors
    return points

def save_ply(points, colors, filename):
    # Salva nuvem de pontos em formato PLY
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            r, g, b = (c * 255).astype(int)
            f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b}\n")

def main():
    if len(sys.argv) != 4:
        print("Uso: python visualize.py path_to_refimgdir path_images_dir path_output_dir")
        return
    
    ref_dir = sys.argv[1]
    images_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Carregar referência
    ref_data = loadmat(os.path.join(ref_dir, 'templatedepth.mat'))
    ref_depth = ref_data['depth']
    K = ref_data['K']
    ref_rgb = cv2.imread(os.path.join(ref_dir, 'templatergb.jpg'))
    
    # Nuvem de referência com cores
    ref_points, ref_colors = depth_to_point_cloud(ref_depth, K, ref_rgb)

    # Coletar todos os pontos e cores
    all_points = []
    all_colors = []
    
    # Referência
    all_points.append(ref_points)
    all_colors.append(ref_colors)
    
    # Plotar
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], c=ref_colors, s=1, label='Referência')
    
    # Listar transforms
    transform_files = sorted([f for f in os.listdir(output_dir) if f.startswith('transform_') and f.endswith('.mat')])
    
    for transform_file in transform_files:
        frame_num = transform_file.split('_')[1].split('.')[0]
        rgb_file = f'frame_{frame_num}.jpg'
        depth_file = f'frame_{frame_num}.mat'
        
        # Carregar dados atuais
        curr_rgb = cv2.imread(os.path.join(images_dir, rgb_file))
        curr_data = loadmat(os.path.join(images_dir, depth_file))
        curr_depth = curr_data['depth']
        curr_points, curr_colors = depth_to_point_cloud(curr_depth, K, curr_rgb)
        
        # Carregar transformação
        trans_data = loadmat(os.path.join(output_dir, transform_file))
        R = trans_data['R']
        T = trans_data['T'].flatten()
        
        # Aplicar transformação
        transformed_points = (R @ curr_points.T).T + T
        
        # Plotar com cores
        ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c=curr_colors, s=1, label=f'Frame {frame_num}')
    
        all_points.append(transformed_points)
        all_colors.append(curr_colors)

    # Combinar
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    # Salvar PLY
    ply_path = os.path.join(output_dir, 'combined_cloud.ply')
    save_ply(combined_points, combined_colors, ply_path)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'visualization.png'))

if __name__ == "__main__":
    main()