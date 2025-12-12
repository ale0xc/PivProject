import os
import sys
import cv2
import numpy as np
import scipy.io as sio
import glob
import part1
import argparse


def get_keypoints(input_folder, fname, output_folder):
    # Prepare filenames
    impath = os.path.join(input_folder, fname)
    base_name = os.path.splitext(fname)[0]
    outim_path = os.path.join(output_folder, fname)
    outkp_path = os.path.join(output_folder, f"{base_name}.mat")

    # Load image
    im = cv2.imread(impath)
    imrgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(imrgb, None)
    for k in kp:
        cx, cy = int(k.pt[0]), int(k.pt[1])
        cv2.circle(im, (cx, cy), 2, (0, 0, 255), -1)
    cv2.imwrite(outim_path, im)

    combined = np.concatenate((np.array([k.pt for k in kp], dtype=np.float32).T, des.T), axis=0)
    sio.savemat(outkp_path, {'kp': combined})


def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Iterate over all images in folder
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(exts):
            try:
                combined = get_keypoints(input_folder, fname, output_folder)
            except Exception as e:
                print(f"Failed to process {fname}: {e}")

def warp_and_save(path_ref, path_images_dir, path_output_dir):

    # 1. Load template (reference) image to get its size
    ref_im = cv2.imread(path_ref)
    if ref_im is None: 
        return
    h_ref, w_ref = ref_im.shape[:2]
    
    # 2. Lister les fichiers Homographie générés par part1
    homography_files = sorted(glob.glob(os.path.join(path_output_dir, "homography_*.mat")))
    
    if not homography_files:
        print("No homography files found.")
        return

    warped_dir = os.path.join(path_output_dir, "warped_images")
    if not os.path.exists(warped_dir):
        os.makedirs(warped_dir)

    for h_file in homography_files:
        seq_num = os.path.splitext(os.path.basename(h_file))[0].split('_')[-1]
        found_imgs = glob.glob(os.path.join(path_images_dir, f"*{seq_num}.*"))
        
        if not found_imgs:
            continue
            
        img_path = found_imgs[0] 
        curr_im = cv2.imread(img_path)
        
        try:
            mat_data = sio.loadmat(h_file)
            H = mat_data["H"]
            
            warped = cv2.warpPerspective(curr_im, H, (w_ref, h_ref))
        
            out_name = f"warped_{seq_num}.jpg"
            cv2.imwrite(os.path.join(warped_dir, out_name), warped)
            
        except Exception as e:
            print(f"Warping error {seq_num}: {e}")
            
    print(f"Rectified images saved in : {warped_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pipeline to compute homography between image pairs using SIFT keypoints.")
    parser.add_argument('path_to_refimg', type=str, help="Path to the reference image")
    parser.add_argument('path_images_dir', type=str, help="Folder containing input images")
    parser.add_argument('path_feature_dir', type=str, help="Output folder for feature files (.mat)")
    parser.add_argument('path_output_dir', type=str, help="Output folder for homography results")
    args = parser.parse_args()

    # Create output directories if they don't exist
    if not os.path.exists(args.path_feature_dir):
        os.makedirs(args.path_feature_dir)
    if not os.path.exists(args.path_output_dir):
        os.makedirs(args.path_output_dir)

    print("--- STEP 1 : FEATURES EXTRACTION ---")

    # A. Template Image (Reference)
    print(f"Extraction features template : {args.path_to_refimg}")
    ref_dir = os.path.dirname(args.path_to_refimg)
    ref_name = os.path.basename(args.path_to_refimg)
    get_keypoints(ref_dir, ref_name, args.path_feature_dir)

    # B. Other Images in Folder
    print(f"Extraction features sequence : {args.path_images_dir}")
    process_folder(args.path_images_dir, args.path_feature_dir)

    print("\n--- STEP 2 : PROCESS PART1 ---")
    
    part1.part1(
        args.path_to_refimg,
        args.path_images_dir,
        args.path_feature_dir,
        args.path_output_dir
    )

    print("\n--- STEP 3 : WARP AND SAVE IMAGES ---")
    warp_and_save(args.path_to_refimg, args.path_images_dir, args.path_output_dir)
    
    print("\n--- PROCESS ENDED ---")