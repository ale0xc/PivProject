import cv2
import scipy.io as sio
import os
import numpy as np

# Configuração
path_temp = "Datasets/Taag"
path_imgs = "Datasets/Taag/sequence"
path_out = "Datasets/Taag/sequence_homographies"
template_path = os.path.join(path_temp, "templateTaag.jpg") # O teu template

# Ler o tamanho do template (para saber o tamanho da imagem de saída)
ref_img = cv2.imread(template_path)
h_ref, w_ref = ref_img.shape[:2]

# Listar resultados
mat_files = sorted([f for f in os.listdir(path_out) if f.endswith(".mat")])

print("Pressiona qualquer tecla para avançar para a próxima imagem. 'q' para sair.")

for mat_file in mat_files:
    # 1. Descobrir qual é a imagem correspondente
    # O nome do ficheiro output é homography_NNNN.mat
    # O nome da imagem original é somename_NNNN.jpg
    # Tens de ajustar este parsing consoante os teus nomes exatos
    suffix = mat_file.split('_')[-1].replace('.mat', '')
    
    # Procura a imagem que tem este sufixo/número
    img_name = [f for f in os.listdir(path_imgs) if suffix in f and f.endswith(".jpg")]
    if not img_name: continue
    img_name = img_name[0]
    
    # 2. Carregar Imagem e Homografia
    img = cv2.imread(os.path.join(path_imgs, img_name))
    data = sio.loadmat(os.path.join(path_out, mat_file))
    H = data["H"]

    # 3. Aplicar a transformação (Warp)
    # Isto simula o que um scanner faria
    warped_img = cv2.warpPerspective(img, H, (w_ref, h_ref))

    # 4. Mostrar Lado a Lado
    # Redimensionar para caber no ecrã se for muito grande
    display_h = 400
    scale = display_h / h_ref
    
    img_s = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    warp_s = cv2.resize(warped_img, (int(w_ref*scale), int(h_ref*scale)))
    
    cv2.imshow("Original (Input)", img_s)
    cv2.imshow("Retificada (Output)", warp_s)
    
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()