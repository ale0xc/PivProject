"""
Generate analysis data and figures for the PIV report.
"""
import numpy as np
import scipy.io as sio
import scipy.spatial.distance as dist
import cv2
import os
import matplotlib.pyplot as plt

def analyze_homography_quality(path_homographies, path_feats, path_template, path_sequence, output_dir):
    """
    Analyse complète des homographies avec métriques correctes.
    """
    # Charger les features du template
    template_data = sio.loadmat(os.path.join(path_feats, "template_features.mat"))
    kp_ref = template_data["keypoints"]
    desc_ref = template_data["descriptors"].astype(np.float32)
    
    # Template image pour reprojection des coins
    img_template = cv2.imread(path_template)
    h_template, w_template = img_template.shape[:2]
    
    # Coins du template
    corners_template = np.array([
        [0, 0],
        [w_template, 0],
        [w_template, h_template],
        [0, h_template]
    ], dtype=np.float32)
    
    h_files = sorted([f for f in os.listdir(path_homographies) if f.endswith('.mat') and f.startswith('homography')])
    
    results = {
        'frames': [],
        'raw_matches': [],
        'ransac_inliers': [],
        'inlier_ratio': [],
        'reprojection_error': [],
        'det_H': [],
    }
    
    for h_file in h_files:
        frame_num = h_file.split('_')[-1].replace('.mat', '')
        
        # Charger homographie
        h_data = sio.loadmat(os.path.join(path_homographies, h_file))
        H = h_data['H']
        
        # Charger features du frame
        feat_file = f"taag_{frame_num}.mat"
        feat_path = os.path.join(path_feats, feat_file)
        
        if not os.path.exists(feat_path):
            continue
            
        frame_data = sio.loadmat(feat_path)
        kp_frame = frame_data["keypoints"]
        desc_frame = frame_data["descriptors"].astype(np.float32)
        
        # Matching avec ratio test
        dists = dist.cdist(desc_frame, desc_ref, metric='euclidean')
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
        
        raw_matches = len(matches_src)
        
        # Calculer les inliers avec H estimée
        if len(matches_src) >= 4:
            ones = np.ones((len(matches_src), 1))
            src_h = np.hstack((matches_src, ones))
            projected = (H @ src_h.T).T
            
            with np.errstate(divide='ignore', invalid='ignore'):
                projected = projected[:, :2] / projected[:, 2:]
            
            errors = np.linalg.norm(projected - matches_dst, axis=1)
            inliers = np.sum(errors < 3.0)  # Seuil de 3 pixels
            
            # Erreur moyenne des inliers
            inlier_errors = errors[errors < 3.0]
            mean_error = np.mean(inlier_errors) if len(inlier_errors) > 0 else 0
            
            results['frames'].append(frame_num)
            results['raw_matches'].append(raw_matches)
            results['ransac_inliers'].append(inliers)
            results['inlier_ratio'].append(inliers / raw_matches * 100 if raw_matches > 0 else 0)
            results['reprojection_error'].append(mean_error)
            results['det_H'].append(np.linalg.det(H))
    
    return results

def generate_latex_results_table(results):
    """Génère un tableau LaTeX propre pour les résultats Part 1."""
    
    latex = """\\begin{table}[H]
\\centering
\\caption{Part 1: Homography Estimation Results per Frame}
\\label{tab:part1_results}
\\small
\\begin{tabular}{|c|c|c|c|c|}
\\hline
\\textbf{Frame} & \\textbf{Raw Matches} & \\textbf{Inliers} & \\textbf{Inlier \\%} & \\textbf{Repr. Error (px)} \\\\
\\hline
"""
    
    for i, frame in enumerate(results['frames']):
        raw = results['raw_matches'][i]
        inl = results['ransac_inliers'][i]
        ratio = results['inlier_ratio'][i]
        err = results['reprojection_error'][i]
        latex += f"{frame} & {raw} & {inl} & {ratio:.1f}\\% & {err:.2f} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\textbf{Mean} & {:.0f} & {:.0f} & {:.1f}\\% & {:.2f} \\\\\n".format(
        np.mean(results['raw_matches']),
        np.mean(results['ransac_inliers']),
        np.mean(results['inlier_ratio']),
        np.mean(results['reprojection_error'])
    )
    latex += "\\hline\n"
    latex += """\\end{tabular}
\\end{table}
"""
    return latex

def generate_summary_table(results):
    """Génère un tableau résumé."""
    
    latex = """\\begin{table}[H]
\\centering
\\caption{Part 1: Summary Statistics}
\\label{tab:part1_summary}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Metric} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} \\\\
\\hline
"""
    
    metrics = [
        ('Raw Matches', 'raw_matches'),
        ('RANSAC Inliers', 'ransac_inliers'),
        ('Inlier Ratio (\\%)', 'inlier_ratio'),
        ('Repr. Error (px)', 'reprojection_error')
    ]
    
    for name, key in metrics:
        data = results[key]
        latex += f"{name} & {np.mean(data):.1f} & {np.std(data):.1f} & {np.min(data):.1f} & {np.max(data):.1f} \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return latex

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    path_homographies = os.path.join(base_dir, "Datasets", "Taag", "sequence_homographies")
    path_feats = os.path.join(base_dir, "Datasets", "Taag", "sequence_features")
    path_template = os.path.join(base_dir, "Datasets", "Taag", "templateTaag.jpg")
    path_sequence = os.path.join(base_dir, "Datasets", "Taag", "sequence")
    output_dir = os.path.join(base_dir, "report", "images")
    
    print("Analysing Part 1 results...")
    results = analyze_homography_quality(path_homographies, path_feats, path_template, path_sequence, output_dir)
    
    print("\n" + "="*70)
    print("PART 1 - HOMOGRAPHY ESTIMATION STATISTICS")
    print("="*70)
    print(f"\nTotal frames processed: {len(results['frames'])}")
    print(f"\nPer-frame breakdown:")
    print("-"*70)
    print(f"{'Frame':<8} {'Raw':<10} {'Inliers':<10} {'Ratio':<12} {'Error (px)':<12}")
    print("-"*70)
    
    for i, frame in enumerate(results['frames']):
        print(f"{frame:<8} {results['raw_matches'][i]:<10} {results['ransac_inliers'][i]:<10} {results['inlier_ratio'][i]:<12.1f}% {results['reprojection_error'][i]:<12.2f}")
    
    print("-"*70)
    print(f"{'MEAN':<8} {np.mean(results['raw_matches']):<10.0f} {np.mean(results['ransac_inliers']):<10.0f} {np.mean(results['inlier_ratio']):<12.1f}% {np.mean(results['reprojection_error']):<12.2f}")
    print("="*70)
    
    # Générer les tableaux LaTeX
    latex_detailed = generate_latex_results_table(results)
    latex_summary = generate_summary_table(results)
    
    print("\n\nLaTeX Detailed Table:")
    print(latex_detailed)
    
    print("\nLaTeX Summary Table:")
    print(latex_summary)
    
    # Sauvegarder
    with open(os.path.join(base_dir, "report", "part1_tables.tex"), "w") as f:
        f.write("% Part 1 Generated Tables\n\n")
        f.write(latex_summary)
        f.write("\n")
        f.write(latex_detailed)
    
    print(f"\nTables saved to {os.path.join(base_dir, 'report', 'part1_tables.tex')}")
