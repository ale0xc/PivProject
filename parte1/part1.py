import os
import numpy as np
import scipy.io as sio

def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)

    T = np.array([
        [1/std, 0, -mean[0]/std],
        [0, 1/std, -mean[1]/std],
        [0, 0, 1]
    ])

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_n = (T @ pts_h.T).T

    return pts_n[:, :2], T

def dlt_homography(p1, p2):
    p1n, T1 = normalize_points(p1)
    p2n, T2 = normalize_points(p2)

    N = p1.shape[0]
    A = []

    for i in range(N):
        x, y = p1n[i]
        u, v = p2n[i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T2) @ Hn @ T1
    return H / H[2, 2]

def ransac(p1, p2, n_iter=567, thresh=3.0):
    best_inliers = []

    N = p1.shape[0]

    for _ in range(n_iter):
        idx = np.random.choice(N, 4, replace=False)
        H = dlt_homography(p1[idx], p2[idx])

        p1h = np.hstack([p1, np.ones((N, 1))])
        proj = (H @ p1h.T).T
        z = proj[:, 2]

        valid = np.abs(z) > 1e-10
        proj_xy = np.zeros((N, 2))
        proj_xy[valid] = proj[valid, :2] / z[valid, None]

        err = np.full(N, np.inf)
        err[valid] = np.linalg.norm(proj_xy[valid] - p2[valid], axis=1)
        inliers = np.where(err < thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    return dlt_homography(p1[best_inliers], p2[best_inliers])

def match_descriptors(desc1, desc2, ratio=0.75):
    matches = []

    for i, d in enumerate(desc1):
        dists = np.linalg.norm(desc2 - d, axis=1)
        idx = np.argsort(dists)

        if dists[idx[0]] < ratio * dists[idx[1]]:
            matches.append((i, idx[0]))

    return matches

def part1(path_to_refimg, path_images_dir, path_feature_dir, path_output_dir):

    os.makedirs(path_output_dir, exist_ok=True)

    ref_name = os.path.splitext(os.path.basename(path_to_refimg))[0]
    ref_data = sio.loadmat(os.path.join(path_feature_dir, ref_name + ".mat"))["kp"]

    ref_pts = ref_data[:2].T
    ref_desc = ref_data[2:].T

    files = sorted([f for f in os.listdir(path_feature_dir) if f.endswith(".mat")])

    for f in files:
        if f.startswith(ref_name):
            continue

        frame_id = f.split("_")[-1].split(".")[0]

        data = sio.loadmat(os.path.join(path_feature_dir, f))["kp"]
        pts = data[:2].T
        desc = data[2:].T

        matches = match_descriptors(desc, ref_desc)

        if len(matches) < 4:
            continue

        p1 = np.array([pts[i] for i, _ in matches])
        p2 = np.array([ref_pts[j] for _, j in matches])

        H = ransac(p1, p2)

        sio.savemat(
            os.path.join(path_output_dir, f"homography_{frame_id}.mat"),
            {"H": H}
        )