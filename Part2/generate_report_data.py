"""
Generate Part 2 analysis data for the PIV report.
"""
import numpy as np
import scipy.io as sio
import os

def analyze_part2_transforms(path_transforms):
    """
    Analyse les transformations 3D estimées.
    """
    t_files = sorted([f for f in os.listdir(path_transforms) if f.startswith('transform_') and f.endswith('.mat')])
    
    results = {
        'frames': [],
        'rotation_magnitude': [],
        'translation_magnitude': [],
        'det_R': [],
        'orthogonality_error': [],
    }
    
    cumulative_R = np.eye(3)
    cumulative_T = np.zeros((3, 1))
    trajectory = [(0, 0, 0)]  # Initial position
    
    for t_file in t_files:
        frame_num = t_file.split('_')[-1].replace('.mat', '')
        
        data = sio.loadmat(os.path.join(path_transforms, t_file))
        R = data['R']
        T = data['T']
        
        # Magnitude de rotation (angle en degrés)
        trace_R = np.trace(R)
        cos_angle = (trace_R - 1) / 2
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        # Magnitude de translation
        trans_mag = np.linalg.norm(T)
        
        # Vérification orthogonalité
        det_R = np.linalg.det(R)
        orth_error = np.linalg.norm(R @ R.T - np.eye(3))
        
        # Trajectoire cumulative
        cumulative_T = cumulative_R @ T + cumulative_T
        cumulative_R = R @ cumulative_R
        
        trajectory.append((cumulative_T[0, 0], cumulative_T[1, 0], cumulative_T[2, 0]))
        
        results['frames'].append(frame_num)
        results['rotation_magnitude'].append(angle_deg)
        results['translation_magnitude'].append(trans_mag)
        results['det_R'].append(det_R)
        results['orthogonality_error'].append(orth_error)
    
    results['trajectory'] = trajectory
    return results

def generate_latex_part2_table(results):
    """Génère un tableau LaTeX pour Part 2."""
    
    latex = """\\begin{table}[H]
\\centering
\\caption{Part 2: Rigid 3D Transformation Results}
\\label{tab:part2_results}
\\small
\\begin{tabular}{|c|c|c|c|c|}
\\hline
\\textbf{Frame} & \\textbf{Rot. (deg)} & \\textbf{Trans. (m)} & \\textbf{det(R)} & \\textbf{Orth. Err.} \\\\
\\hline
"""
    
    for i, frame in enumerate(results['frames']):
        rot = results['rotation_magnitude'][i]
        trans = results['translation_magnitude'][i]
        det = results['det_R'][i]
        orth = results['orthogonality_error'][i]
        latex += f"{frame} & {rot:.2f} & {trans:.4f} & {det:.6f} & {orth:.2e} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\textbf{Mean} & {:.2f} & {:.4f} & {:.6f} & {:.2e} \\\\\n".format(
        np.mean(results['rotation_magnitude']),
        np.mean(results['translation_magnitude']),
        np.mean(results['det_R']),
        np.mean(results['orthogonality_error'])
    )
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    return latex

def generate_part2_summary(results):
    """Génère un résumé pour Part 2."""
    
    latex = """\\begin{table}[H]
\\centering
\\caption{Part 2: Summary Statistics}
\\label{tab:part2_summary}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Metric} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} \\\\
\\hline
"""
    
    metrics = [
        ('Rotation Magnitude (deg)', 'rotation_magnitude'),
        ('Translation Magnitude (m)', 'translation_magnitude'),
        ('det(R)', 'det_R'),
        ('Orthogonality Error', 'orthogonality_error')
    ]
    
    for name, key in metrics:
        data = results[key]
        if key == 'orthogonality_error':
            latex += f"{name} & {np.mean(data):.2e} & {np.std(data):.2e} & {np.min(data):.2e} & {np.max(data):.2e} \\\\\n"
        else:
            latex += f"{name} & {np.mean(data):.4f} & {np.std(data):.4f} & {np.min(data):.4f} & {np.max(data):.4f} \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return latex

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_transforms = os.path.join(script_dir, "results")
    
    print("Analysing Part 2 results...")
    results = analyze_part2_transforms(path_transforms)
    
    print("\n" + "="*70)
    print("PART 2 - 3D RIGID TRANSFORMATION STATISTICS")
    print("="*70)
    print(f"\nTotal frames processed: {len(results['frames'])}")
    print(f"\nPer-frame breakdown:")
    print("-"*70)
    print(f"{'Frame':<8} {'Rot (deg)':<12} {'Trans (m)':<12} {'det(R)':<12} {'Orth. Err':<12}")
    print("-"*70)
    
    for i, frame in enumerate(results['frames']):
        print(f"{frame:<8} {results['rotation_magnitude'][i]:<12.2f} {results['translation_magnitude'][i]:<12.4f} {results['det_R'][i]:<12.6f} {results['orthogonality_error'][i]:<12.2e}")
    
    print("-"*70)
    print(f"{'MEAN':<8} {np.mean(results['rotation_magnitude']):<12.2f} {np.mean(results['translation_magnitude']):<12.4f} {np.mean(results['det_R']):<12.6f} {np.mean(results['orthogonality_error']):<12.2e}")
    print("="*70)
    
    # Générer les tableaux LaTeX
    latex_detailed = generate_latex_part2_table(results)
    latex_summary = generate_part2_summary(results)
    
    print("\n\nLaTeX Detailed Table:")
    print(latex_detailed)
    
    print("\nLaTeX Summary Table:")
    print(latex_summary)
    
    # Sauvegarder
    base_dir = os.path.dirname(script_dir)
    with open(os.path.join(base_dir, "report", "part2_tables.tex"), "w") as f:
        f.write("% Part 2 Generated Tables\n\n")
        f.write(latex_summary)
        f.write("\n")
        f.write(latex_detailed)
    
    print(f"\nTables saved to {os.path.join(base_dir, 'report', 'part2_tables.tex')}")
