import os
import cv2
import numpy as np
import scipy.io as sio
import argparse
import part1

def extract_and_save_features(template_path, feature_dir):
    os.makedirs(feature_dir, exist_ok=True)

    # Prepare filenames
    fname = os.path.basename(template_path)
    name, _ = os.path.splitext(fname)

    # Load image
    img = cv2.imread(template_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(img_rgb, None)
    for k in kp:
        cx, cy = int(k.pt[0]), int(k.pt[1])
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(feature_dir, fname), img)

    pts = np.array([k.pt for k in kp], dtype=np.float32).T  # (2, N)
    combined = np.vstack((pts, des.T))

    out_path = os.path.join(feature_dir, f"{name}.mat")
    sio.savemat(out_path, {"kp": combined})


def process_images(images_dir, features_dir):
    # Supported image extensions
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith(exts)]
    )

    for fname in files:
        img_path = os.path.join(images_dir, fname)
        extract_and_save_features(img_path, features_dir)

def warp_and_save(image_path, H, ref_shape, output_path):
    im = cv2.imread(image_path)
    warped = cv2.warpPerspective(im, H, (ref_shape[1], ref_shape[0]))
    cv2.imwrite(output_path, warped)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIV Part 1 - Main")
    parser.add_argument("path_to_refimg", help="Path to reference image")
    parser.add_argument("path_images_dir", help="Path to image sequence directory")
    parser.add_argument("path_feature_dir", help="Path to feature directory")
    parser.add_argument("path_output_dir", help="Path to output directory")

    args = parser.parse_args()

    extract_and_save_features(args.path_to_refimg, args.path_feature_dir)

    process_images(args.path_images_dir, args.path_feature_dir)

    part1.part1(
        args.path_to_refimg,
        args.path_images_dir,
        args.path_feature_dir,
        args.path_output_dir
    )

    ref_img = cv2.imread(args.path_to_refimg)
    h_ref, w_ref = ref_img.shape[:2]

    for fname in sorted(os.listdir(args.path_images_dir)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        frame_id = os.path.splitext(fname)[0].split("_")[-1]
        H_path = os.path.join(args.path_output_dir, f"homography_{frame_id}.mat")

        if not os.path.exists(H_path):
            continue

        H = sio.loadmat(H_path)["H"]

        img_path = os.path.join(args.path_images_dir, fname)
        out_warp = os.path.join(
            args.path_output_dir,
            f"warp_{frame_id}.jpg"
        )

        warp_and_save(img_path, H, ref_img.shape, out_warp)