import cv2
import numpy as np
import os
import scipy.io as sio
import sys
from part1 import part1  # Importa a função que vamos criar a seguir

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- CONFIGURAÇÃO ---
# Caminhos relativos (ajusta se necessário)
script_dir = os.path.dirname(os.path.abspath(__file__))
path_imgs = os.path.join(script_dir, "imagens")       # Pasta onde estão as imagens jpg
path_feats = os.path.join(script_dir, "features")     # Pasta onde vamos guardar os .mat temporários
path_out = os.path.join(script_dir, "output")         # Pasta para o resultado final
template_name = "20251028_170011.jpg" # O teu template específico

# Verificar argumentos da linha de comando (conforme PDF [cite: 79])
if len(sys.argv) > 4:
    ref_img_path = sys.argv[1]
    path_imgs = sys.argv[2]
    path_feats = sys.argv[3]
    path_out = sys.argv[4]
else:
    # Fallback para execução simples sem argumentos
    ref_img_path = os.path.join(path_imgs, template_name)

ensure_dir(path_feats)
ensure_dir(path_out)

# 1. Processar o Template
print(f"A processar template: {ref_img_path}")
img_ref = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)

if img_ref is None:
    print(f"Erro: Não foi possível ler a imagem do template: {ref_img_path}")
    print("Verifique se o caminho está correto e se o ficheiro existe.")
    sys.exit(1)

sift = cv2.SIFT_create() # Podes usar ORB se o SIFT for lento, mas SIFT é melhor
kp_ref, desc_ref = sift.detectAndCompute(img_ref, None)

# Guardar features do template (necessário para o matching na part1)
# O formato dos ficheiros segue o padrão somename_NNNN.mat [cite: 81]
# Mas o template precisa de um nome fixo para a part1 o encontrar.
sio.savemat(os.path.join(path_feats, "template_features.mat"), {
    "keypoints": cv2.KeyPoint_convert(kp_ref), # Guarda apenas coordenadas (N, 2)
    "descriptors": desc_ref
})

# 2. Processar a Sequência
files = sorted([f for f in os.listdir(path_imgs) if f.endswith(".jpg") and f != template_name])

for f in files:
    img_path = os.path.join(path_imgs, f)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    kp, desc = sift.detectAndCompute(img, None)
    
    if kp is None or desc is None:
        print(f"Aviso: Nenhuma feature encontrada em {f}")
        continue

    # O nome do ficheiro .mat deve corresponder ao da imagem
    mat_name = f.replace(".jpg", ".mat")
    save_path = os.path.join(path_feats, mat_name)
    
    sio.savemat(save_path, {
        "keypoints": cv2.KeyPoint_convert(kp),
        "descriptors": desc
    })
    print(f"Features extraídas: {f}")

print("--- Extração concluída. A iniciar Part1 ---")

# Chama a função part1 conforme o PDF [cite: 89, 94]
part1(ref_img_path, path_imgs, path_feats, path_out)