import numpy as np
import cv2

from utils.linalg import find_null_space, skew

def equation_matrix(p1, p2):
    '''
    Given two sets of points x1, x2, return the correspondance matrix A
    Each row in the A matrix is constructed as
    [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    keeping in mind that X'.T * F * X = 0
    '''
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]
    
    A = np.array([
        p1x * p2x, p1x * p2y, p1x, p1y * p2x, p1y * p2y, p1y, p2x, p2y, np.ones(len(p1x))
    ]).T   
    return A
    
def scale_points(points):
    '''
    Scale the points so that the centroid of the points is at the origin and the average
    distance to the origin is equal to sqrt(2). Refer to Hartley and Zisserman for more details.
    '''
    assert points.shape[0] == 3
    
    mean = points[:2].mean(axis=1)
    distances = np.sqrt(np.sum((points[:2] - mean[:, None]) ** 2, axis=0))
    scale = np.sqrt(2) / np.mean(distances)
    
    scale_matrix = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    adjusted_points = np.dot(scale_matrix, points)
    
    return adjusted_points, scale_matrix

def compute_fundamental_or_essential_matrix(x1, x2, compute_essential=False, intrinsics1=None, intrinsics2=None):
    '''
    Compute the fundamental or essential matrix given 3xN homogeneous corresponding points 
    x1, x2 using the 8-point algorithm.
    '''
    assert x1.shape == x2.shape
    assert x1.shape[0] == 3
    assert x2.shape[0] == 3
    
    # Scale the points
    x1, scale_matrix1 = scale_points(x1)
    x2, scale_matrix2 = scale_points(x2)
    
    A = equation_matrix(x1, x2)
    F = find_null_space(A).reshape(3, 3)
    
    # Make F rank 2 by zeroing out the smallest singular value
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(U, np.dot(np.diag(S), Vt))
    
    if compute_essential:
        if intrinsics1 is None and intrinsics2 is None:
            print("Intrinsics not provided. Cannot compute essential matrix.")
            exit(1)
        
        F = np.dot(intrinsics1.T, np.dot(F, intrinsics2))
    
        U, S, Vt = np.linalg.svd(F)
        S = np.diag([1, 1, 0])
        F = np.dot(U, np.dot(S, Vt))
    else: 
        S = np.diag(S)
        F = np.dot(U, np.dot(S, Vt))
    
    # Reverse the scaling. We know that p1.T * F * p2 = 0
    F = np.dot(scale_matrix1.T, np.dot(F, scale_matrix2))
    return F/F[2, 2]

# TODO behavior is not as expected
def compute_f_or_e_ransac(x1, x2, compute_essential=False, intrinsics1=None, intrinsics2=None, threshold=0.1, iterations=500):
    '''
    Compute the fundamental or essential matrix using RANSAC
    '''
    assert x1.shape == x2.shape
    assert x1.shape[0] == 3
    assert x2.shape[0] == 3
            
    for i in range(iterations):
        # Randomly sample 8 points
        indices = np.random.choice(x1.shape[1], 8, replace=False)
        F = compute_fundamental_or_essential_matrix(x1[:, indices], x2[:, indices], compute_essential, intrinsics1, intrinsics2)
        
        # Calculate the error
        error = np.abs(np.diag(np.dot(x1.T, np.dot(F, x2))))
        inliers = error < threshold
        
        if np.sum(inliers) > 0.7 * x1.shape[1]:
            print("Computed F/E using RANSAC")
            return F
    
    print("RANSAC failed to find F/E")
    return compute_fundamental_or_essential_matrix(x1, x2, compute_essential, intrinsics1, intrinsics2)

def find_sift_correspondences(img1, img2, fit_homography=True):
    '''
    Given two images, find SIFT correspondences between them. 
    Optionally, fit a homography to the matches. The output is a tuple of two 2xN arrays.
    '''
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    # Find point matches
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's SIFT matching ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

    if fit_homography:
        # Constrain matches to fit homography
        retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
        mask = mask.ravel()
    
        # We select only inlier points
        src_pts = src_pts[mask == 1]
        dst_pts = dst_pts[mask == 1]

    return src_pts.T, dst_pts.T

def track_features(img1, img2, points):
    """
    Track features between two images
    """
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, points, None, **lk_params)
    return new_points, status

def triangulate_point(pt1, pt2, m1, m2):
    """
    Triangulation using DLT
    pt1 and m1 * X are parallel and cross product = 0
    pt1 x m1 * X  =  pt2 x m2 * X  =  0
    """
    A = np.vstack([
        np.dot(skew(pt1), m1),
        np.dot(skew(pt2), m2)
    ])
    P = find_null_space(A)
    return P / P[3]

def linear_triangulation(p1, p2, m1, m2):
    """
    Linear triangulation to find the 3D point X
    where p1 = m1 * X and p2 = m2 * X. Solve AX = 0.
    :param p1, p2: 2D points in homo. or catesian coordinates. Shape (3 x n)
    :param m1, m2: Camera matrices associated with p1 and p2. Shape (3 x 4)
    :returns Homogenous 3d triangulated points. shape(4 x n)
    """
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    # Single correspondance wise triangulation
    for i in range(num_points):
        res[:, i] = triangulate_point(p1[:, i], p2[:, i], m1, m2)

    return res

def compute_P_from_essential(E):
    """ 
    Compute the second camera matrix from an essential matrix 
    E = [t]R (assuming P1 = [I 0])
    :returns list of 4 possible camera matrices.
    """
    U, _, V = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # Create 4 possible camera matrices (Hartley, Zisserman)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [
        np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
        np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
        np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
        np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T
    ]

    return P2s
    
def compute_P2_from_P1(E, P1, point1n, point2n):
    """
    Given the essential matrix E, the first camera matrix P1
    and a normalized point correspondence, calculate the second 
    camera matrix P2  
    """            
    P2s = compute_P_from_essential(E)
    
    ind = -1
    for i, P2 in enumerate(P2s):
        # Find the correct camera parameters
        d1 = triangulate_point(
            point1n, point2n, P1, P2)
    
        # Convert P2 from camera view to world view
        P2_inv = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_inv[:3, :4], d1)
    
        # Check Z is positive, i.e. point is in front of cameras
        if d1[2] > 0 and d2[2] > 0:
            ind = i

    assert ind != -1, 'No valid solution found'

    P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
    return P2
        
def calculate_epipolar_lines(points, E, intrinsics):
    """
    Calculate epipolar lines given the fundamental matrix F
    :param points: 2D points in homogeneous coordinates. Shape (3 x n)
    :param F: Fundamental matrix. Shape (3 x 3)
    :returns Epipolar lines. Shape (3 x n)
    """
    assert points.shape[0] == 3
    F = np.dot(np.linalg.inv(intrinsics.T), np.dot(E, np.linalg.inv(intrinsics)))
    return np.dot(F, points)

def normalized_cross_correlation(patch1, patch2):
    """
    Calculate the normalized cross correlation between two image patches
    """
    patch1_norm = (patch1 - np.mean(patch1)) / np.std(patch1)
    patch2_norm = (patch2 - np.mean(patch2)) / np.std(patch2)
    return np.sum(patch1_norm*patch2_norm) / np.sqrt(np.sum(patch1_norm**2) * np.sum(patch2_norm**2))

# TODO implement gaussian filter
def block_matching(px1, line2, img1, img2, block_size=25, search_window=50, stride=5):
    """
    Perform block matching between two images (slambook2, p. 226)
    :param px1: Pixel coordinate in the first image
    :param line2: Epipolar line in the second image
    :param img1, img2: Images to perform block matching on
    :param block_size: Size of the block
    :param search_window: Size of the search window
    :returns px2: Corresponding pixel coordinate in the second image
    :returns None: If no corresponding point is found
    """
    assert img1.shape == img2.shape
    assert px1.shape == (2,)
    assert line2.shape[0] == 3
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Get pixel coordinates from the epipolar line
    y_vals = np.arange(img2.shape[0])
    x_vals = (-line2[1] * y_vals - line2[2]) / line2[0]
    valid_indices = np.logical_and(x_vals >= 0, x_vals < img2.shape[1])
    y_vals = y_vals[valid_indices]
    x_vals = x_vals[valid_indices]
    
    # Get the block from the first image
    A = img1[int(px1[1] - block_size/2):int(px1[1] + block_size/2), 
             int(px1[0] - block_size/2):int(px1[0] + block_size/2)]
    
    if A.shape != (block_size, block_size):
        return None
        
    # Perform block matching
    max_ccr_score = 0
    max_x = 0
    max_y = 0
    for i in range(0, len(y_vals), stride):
        y_val = y_vals[i]
        x_val = x_vals[i]
                                        
        for u in range(-search_window//2, search_window//2, stride):
            for v in range(-search_window//2, search_window//2, stride):
                y_min = int(y_val - block_size/2 + u)
                y_max = int(y_val + block_size/2 + u)
                x_min = int(x_val - block_size/2 + v)
                x_max = int(x_val + block_size/2 + v)                
                B = img2[y_min:y_max, x_min:x_max]
                
                if B.shape != (block_size, block_size):
                    continue
                    
                ccr_score = normalized_cross_correlation(A, B)
                if ccr_score > max_ccr_score:
                    max_x = x_val + v
                    max_y = y_val + u
                    max_ccr_score = max(ccr_score, max_ccr_score)
    
    if max_ccr_score > 0:
        return np.array([max_x, max_y])
    
    return None
