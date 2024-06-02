import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

import matplotlib
matplotlib.use('TkAgg')

def load_ppm(file):
    '''
    Load a PPM file as a numpy array
    '''
    with open(file, 'rb') as f:
        assert f.readline() == b'P6\n'
        width, height = map(int, f.readline().split())
        assert f.readline() == b'255\n'
        img = np.fromfile(f, dtype=np.uint8).reshape(height, width, 3)
    return img

def find_features(img0, img1):
    """
    Find feature correspondences between two images
    """
    img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(img0gray, None)
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
    
    _, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 100.0)
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]

    return pts0, pts1

def reprojection_error(X, pts, extrinsics, intrinsics):
    """
    Calculate reprojection error
    """
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    r, _ = cv2.Rodrigues(R)
    
    p, _ = cv2.projectPoints(X, r, t, intrinsics, distCoeffs=None)
    
    p = np.float32(p[:, 0, :])
    pts = np.float32(pts)    
    error = cv2.norm(p, pts, cv2.NORM_L2) / len(p)
    return error, p

def PnP(X, pts1, intrinsics, d, pts0):
    """
    Solve PnP problem
    """
    ret, rvecs, t, inliers = cv2.solvePnPRansac(X, pts1, intrinsics, d, cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvecs)

    if inliers is not None:
        pts0 = pts0[inliers[:, 0]]
        X = X[inliers[:, 0]]
        pts1 = pts1[inliers[:, 0]]

    return R, t, X, pts0, pts1

def common_points(pts1, pts2):
    """
    Find common points between two sets of points
    """
    idx1 = []
    idx2 = []
    for i in range(len(pts1)):
        indices = np.where(pts2 == pts1[i])
        if len(indices[0]) > 0:
            idx1.append(i)
            idx2.append(indices[0][0])

    return np.array(idx1), np.array(idx2)

if __name__ == "__main__":
    # dataset_dir = "/home/bharath/Documents/SLAM/dataset/GustavIIAdolf/"
    # image_names = sorted(os.listdir(dataset_dir))
    # img1 = cv2.imread(dataset_dir + image_names[0])
    # img2 = cv2.imread(dataset_dir + image_names[1])

    dataset_dir = "../images/dino/"
    image_names = ['viff.' + str(i).zfill(3) + '.ppm' for i in range(15)]        
    img1 = load_ppm(dataset_dir + image_names[0])
    img2 = load_ppm(dataset_dir + image_names[1])
    
    intrinsics = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])    
    
    # Find feature correspondences between two images
    img_pts1, img_pts2 = find_features(img1, img2)
    
    # Calculate essential matrix with feature correspondences and intrinsics
    E, mask = cv2.findEssentialMat(img_pts1, img_pts2, intrinsics, method=cv2.RANSAC, prob=0.999, threshold=2.0, mask=None)
    img_pts1 = img_pts1[mask.ravel() > 0]
    img_pts2 = img_pts2[mask.ravel() > 0]
    
    # Recover pose from essential matrix
    _, R, t, mask = cv2.recoverPose(E, img_pts1, img_pts2, intrinsics)
    img_pts1 = img_pts1[mask.ravel() > 0]
    img_pts2 = img_pts2[mask.ravel() > 0]
    
    P1 = np.eye(3, 4)
    extrinsics = np.hstack((R, np.expand_dims(np.dot(R, t.ravel()), axis=1)))
    P2 = np.dot(intrinsics, extrinsics)
    
    # Triangulate points given the camera matrices and feature correspondences
    triangulated_points = cv2.triangulatePoints(P1, P2, img_pts1.T, img_pts2.T)
    triangulated_points /= triangulated_points[3]
    
    # Calculate reprojection error
    reprojection_err, reprojected_pts = reprojection_error(triangulated_points[:3].T, img_pts1, extrinsics, intrinsics)
    
    # Solve PnP problem to refine the pose and triangulated points
    inital_extrinsics = np.zeros((5, 1), dtype=np.float32)
    R, t, triangulated_points, img_pts1, img_pts2 = PnP(triangulated_points[:3].T, img_pts2, intrinsics, inital_extrinsics, img_pts1)
    
    # Track poses and triangulated points
    poses = [P1, P2]
    agg_triangulated_points = np.copy(triangulated_points)
    
    Pprevprev = np.copy(P1)
    Pprev = np.copy(P2)
    img_pts_prev = np.copy(img_pts2)
    img_prev = np.copy(img2)
    triangulated_points_prev = np.copy(triangulated_points)
    
    for i in range(2, len(image_names)):
        if i == 15:
            break
        print("-----------------------------")
        print(f"Processing image {i}/{len(image_names)}")
        # imgn = cv2.imread(dataset_dir + image_names[i])
        imgn = load_ppm(dataset_dir + image_names[i])
        
        # Find feature correspondences between two images
        img_pts_, img_ptsn = find_features(img_prev, imgn)
        print(f"Feature points: {len(img_pts_)}")
        
        # Find common points between the previous and current features
        common_idx_prev, common_idx_curr = common_points(img_pts_prev, img_pts_)
        common_img_pts_ = img_pts_[common_idx_curr]
        common_img_ptsn = img_ptsn[common_idx_curr]
        common_triangulated_points = triangulated_points_prev[common_idx_prev]
        print(f"Common points: {len(common_img_ptsn)}")
        
        # Solve PnP problem to recover the pose
        inital_extrinsics = np.zeros((5, 1), dtype=np.float32)
        R, t, triangulated_points, common_img_pts_, common_img_ptsn = PnP(common_triangulated_points, common_img_ptsn, intrinsics, inital_extrinsics, common_img_pts_)    
        extrinsics_new = np.hstack((R, t))
        Pnew = np.dot(intrinsics, extrinsics_new)
        
        # Triangulate points given the camera matrices and feature correspondences
        triangulated_points = cv2.triangulatePoints(Pprev, Pnew, img_pts_.T, img_ptsn.T)
        triangulated_points /= triangulated_points[3]
        print(f"Triangulated points: {triangulated_points.shape}")
        
        # Calculate reprojection error
        reprojection_err, reprojected_pts = reprojection_error(triangulated_points[:3].T, img_ptsn, extrinsics, intrinsics)
        triangulated_points = triangulated_points[:3].T
        print(f"Reprojection error: {reprojection_err}")
        
        # Prepare for next iteration
        Pprev = np.copy(Pnew)
        Pprevprev = np.copy(Pprev)
        img_pts_prev = np.copy(img_ptsn)
        triangulated_points_prev = np.copy(triangulated_points)
        poses.append(Pnew)
        agg_triangulated_points = np.vstack((agg_triangulated_points, triangulated_points))
        img_prev = np.copy(imgn)
    
    # # Display feature correspondences
    # fig, ax = plt.subplots(1, 2)
    # ax[0].autoscale_view('tight')
    # ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # ax[0].plot(img_pts1.T[0], img_pts1.T[1], 'r.')
    # ax[1].autoscale_view('tight')
    # ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    # ax[1].plot(img_pts2.T[0], img_pts2.T[1], 'r.')
    # fig.show()
    
    # Display reconstructed 3D points
    triangulated_points = agg_triangulated_points[:, :3] - np.mean(agg_triangulated_points[:, :3], axis=0)
    triangulated_points = triangulated_points.T

    limit = 0.1
    
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.add_subplot(projection='3d')
    ax.plot(triangulated_points[0], triangulated_points[1], triangulated_points[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.view_init(elev=135, azim=90)
    plt.show()    
