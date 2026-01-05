import numpy as np
import scipy.io as sio
import cv2
import os
from pathlib import Path

def analyze_homographies(path_homographies, path_imgs, path_feats, path_template):
    """
    Analyse les homographies estimées et calcule des statistiques quantitatives.
    Retourne des métriques pour chaque frame.
    """
    
    # Charger les features du template
    template_data = sio.loadmat(os.path.join(path_feats, "template_features.mat"))
    kp_ref = template_data["keypoints"]
    
    # Charger le template
    img_template = cv2.imread(path_template, cv2.IMREAD_GRAYSCALE)
    h_template, w_template = img_template.shape[:2]
    
    # Listeir les fichiers homographiques
    h_files = sorted([f for f in os.listdir(path_homographies) if f.endswith('.mat')])
    
    results = {
        'frames': [],
        'num_inliers': [],
        'reprojection_errors': [],
        'det_H': [],
        'condition_number': []
    }
    
    for h_file in h_files:
        # Extraire le numéro du frame
        frame_num = h_file.split('_')[-1].replace('.mat', '')
        
        # Charger la homographie
        h_data = sio.loadmat(os.path.join(path_homographies, h_file))
        H = h_data['H']
        
        # Charger les features du frame
        frame_feat_file = f"taag_{frame_num}.mat"
        frame_feat_path = os.path.join(path_feats, frame_feat_file)
        
        if not os.path.exists(frame_feat_path):
            continue
            
        frame_data = sio.loadmat(frame_feat_path)
        kp_frame = frame_data["keypoints"]
        
        results['frames'].append(frame_num)
        
        # Calculer l'erreur de reprojection
        if len(kp_frame) > 0:
            ones = np.ones((len(kp_frame), 1))
            kp_h = np.hstack((kp_frame, ones))
            projected = (H @ kp_h.T).T
            projected = projected[:, :2] / projected[:, 2:]
            
            # Erreur en projetant les coins du template
            corners_template = np.array([
                [0, 0],
                [w_template, 0],
                [w_template, h_template],
                [0, h_template]
            ], dtype=np.float32)
            
            ones_corners = np.ones((4, 1))
            corners_h = np.hstack((corners_template, ones_corners))
            projected_corners = (H @ corners_h.T).T
            projected_corners = projected_corners[:, :2] / projected_corners[:, 2:]
            
            # Métriques
            mean_error = np.mean(np.linalg.norm(projected - kp_frame, axis=1))
            det_h = np.linalg.det(H)
            cond_num = np.linalg.cond(H)
            
            results['num_inliers'].append(len(kp_frame))
            results['reprojection_errors'].append(mean_error)
            results['det_H'].append(det_h)
            results['condition_number'].append(cond_num)
    
    return results

def print_statistics(results):
    """Affiche les statistiques de manière lisible."""
    
    print("\n" + "="*70)
    print("STATISTIQUES PART1 - HOMOGRAPHIES")
    print("="*70)
    
    if not results['frames']:
        print("Aucun résultat trouvé.")
        return
    
    print(f"\nNombre de frames traités: {len(results['frames'])}")
    print(f"\nPar frame:")
    print("-"*70)
    print(f"{'Frame':<8} {'Inliers':<12} {'Repr. Error (px)':<20} {'det(H)':<15}")
    print("-"*70)
    
    for i, frame in enumerate(results['frames']):
        inliers = results['num_inliers'][i]
        error = results['reprojection_errors'][i]
        det_h = results['det_H'][i]
        print(f"{frame:<8} {inliers:<12} {error:<20.4f} {det_h:<15.4f}")
    
    print("\n" + "-"*70)
    print(f"{'MOYENNE':<8} {np.mean(results['num_inliers']):<12.1f} {np.mean(results['reprojection_errors']):<20.4f}")
    print(f"{'ÉCART-TYPE':<8} {np.std(results['num_inliers']):<12.1f} {np.std(results['reprojection_errors']):<20.4f}")
    print(f"{'MIN':<8} {np.min(results['num_inliers']):<12} {np.min(results['reprojection_errors']):<20.4f}")
    print(f"{'MAX':<8} {np.max(results['num_inliers']):<12} {np.max(results['reprojection_errors']):<20.4f}")
    print("="*70 + "\n")

def generate_latex_table(results):
    """Génère un tableau LaTeX avec les résultats."""
    
    latex = "\\begin{table}[H]\n"
    latex += "\\centering\n"
    latex += "\\small\n"
    latex += "\\begin{tabular}{|c|r|r|r|}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Frame} & \\textbf{Inliers} & \\textbf{Repr. Error (px)} & \\textbf{det(H)} \\\\\n"
    latex += "\\hline\n"
    
    for i, frame in enumerate(results['frames']):
        inliers = results['num_inliers'][i]
        error = results['reprojection_errors'][i]
        det_h = results['det_H'][i]
        latex += f"{frame} & {inliers} & {error:.4f} & {det_h:.4f} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\textbf{Mean} & {:.1f} & {:.4f} & {:.4f} \\\\\n".format(
        np.mean(results['num_inliers']),
        np.mean(results['reprojection_errors']),
        np.mean(results['det_H'])
    )
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Homography estimation results per frame.}\n"
    latex += "\\label{tab:part1_results}\n"
    latex += "\\end{table}\n"
    
    return latex

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
path_homographies = os.path.join(os.path.dirname(script_dir), "Datasets", "Taag", "sequence_homographies")
path_imgs = os.path.join(os.path.dirname(script_dir), "Datasets", "Taag", "sequence")
path_feats = os.path.join(os.path.dirname(script_dir), "Datasets", "Taag", "sequence_features")
path_template = os.path.join(os.path.dirname(script_dir), "Datasets", "Taag", "templateTaag.jpg")

if __name__ == "__main__":
    results = analyze_homographies(path_homographies, path_imgs, path_feats, path_template)
    print_statistics(results)
    
    # Générer tableau LaTeX
    latex_table = generate_latex_table(results)
    print("Tableau LaTeX généré:")
    print(latex_table)
    
    # Sauvegarder dans un fichier
    with open("part1_statistics.txt", "w") as f:
        f.write("STATISTIQUES PART1\n\n")
        f.write(latex_table)
    print("Tableau sauvegardé dans 'part1_statistics.txt'")
