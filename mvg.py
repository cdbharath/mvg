import numpy as np
import cv2

from linalg import find_null_space, skew

# TODO Check how this works
def correspondance_matrix(p1, p2):
    '''
    Given two sets of points x1, x2, return the correspondance matrix A
    Each row in the A matrix is constructed as
    [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
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

def compute_fundamental_or_essential_matrix(x1, x2, compute_essential=False):
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
    
    A = correspondance_matrix(x1, x2)
    F = find_null_space(A).reshape(3, 3)
    
    # Make F rank 2 by zeroing out the smallest singular value
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    if compute_essential:
        S = np.diag([1, 1, 0])
    F = np.dot(U, np.dot(S, Vt))
    
    # Reverse the scaling. We know that p1.T * F * p2 = 0
    F = np.dot(scale_matrix1.T, np.dot(F, scale_matrix2))
    return F/F[2, 2]


def find_sift_correspondences(img1, img2, fit_homography=True):
    '''
    Given two images, find SIFT correspondences between them. 
    Optionally, fit a homography to the matches. The output is a tuple of two 2xN arrays.
    '''
    sift = cv2.xfeatures2d.SIFT_create()

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
        if m.distance < 0.8 * n.distance:
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
    U, S, V = np.linalg.svd(E)

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
    