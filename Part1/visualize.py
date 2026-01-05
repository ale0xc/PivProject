import cv2
import scipy.io as sio
import os
import numpy as np

# Configuração
path_imgs = "Datasets/Taag/sequence"
path_out = "Datasets/Taag/sequence_homographies"
template_path = "Datasets/Taag/templateTaag.jpg"

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

    # 4. Salvar Lado a Lado
    # Redimensionar para caber no ecrã se for muito grande
    display_h = 400
    scale = display_h / h_ref
    
    img_s = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    warp_s = cv2.resize(warped_img, (int(w_ref*scale), int(h_ref*scale)))
    
    # Garantir que as duas imagens têm a mesma altura para colocá-las lado a lado
    max_h = max(img_s.shape[0], warp_s.shape[0])
    
    # Adicionar padding se necessário
    img_s_padded = cv2.copyMakeBorder(img_s, 0, max_h - img_s.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    warp_s_padded = cv2.copyMakeBorder(warp_s, 0, max_h - warp_s.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Concatenar lado a lado
    combined = cv2.hconcat([img_s_padded, warp_s_padded])
    
    # Criar diretório de saída se não existir
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Salvar a imagem combinada
    output_name = f"{output_dir}/combined_{suffix}.jpg"
    cv2.imwrite(output_name, combined)
    print(f"Salvo: {output_name}")
    
    cv2.imshow("Original (Input) | Retificada (Output)", combined)
    
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()