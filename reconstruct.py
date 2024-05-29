import numpy as np
import scipy
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import cv2

from utils.mvg import find_sift_correspondences, compute_fundamental_or_essential_matrix, \
    compute_P2_from_P1, linear_triangulation, compute_f_or_e_ransac, track_features
from utils.transformations import euclidean_to_homogeneous
from utils.linalg import decompose_projection_matrix

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
    
def reconstruct(img1_path, img2_path, intrinsics1, intrinsics2):
    '''
    Reconstruct 3D points from two images
    
    @param img1_path: path to the first image
    @param img2_path: path to the second image
    @param intrinsic: intrinsic camera matrix
    '''
    img1 = load_ppm(img1_path)
    img2 = load_ppm(img2_path)
    
    pts1, pts2 = find_sift_correspondences(img1, img2)
    points1 = euclidean_to_homogeneous(pts1.T).T
    points2 = euclidean_to_homogeneous(pts2.T).T
    
    points1n = np.dot(np.linalg.inv(intrinsics1), points1)
    points2n = np.dot(np.linalg.inv(intrinsics2), points2)
    E = compute_f_or_e_ransac(points1n, points2n, compute_essential=True, intrinsics1=intrinsics1, intrinsics2=intrinsics2)
    
    # Assuming the first camera is at the origin
    P1 = np.eye(3, 4)    
    P2 = compute_P2_from_P1(E, P1, points1n[:, 0], points2n[:, 0])
    print(f"Projection matrix 1:\n{P1}")
    print(f"Projection matrix 2:\n{P2}")

    triangulated_points = linear_triangulation(points1n, points2n, P1, P2)    
    return triangulated_points, (img1, img2, points1, points2, E, P1, P2)

# TODO Check why this is not accurate
def reconstruct_with_projection_matrices(img1_path, img2_path, P1, P2):
    '''
    Reconstruct 3D points from two images using projection matrices
    
    @param img1_path: path to the first image
    @param img2_path: path to the second image
    @param P1: projection matrix of the first camera
    @param P2: projection matrix of the second camera
    '''
    img1 = load_ppm(img1_path)
    img2 = load_ppm(img2_path)
    
    pts1, pts2 = find_sift_correspondences(img1, img2)
    
    points1 = euclidean_to_homogeneous(pts1.T).T
    points2 = euclidean_to_homogeneous(pts2.T).T
    
    points1n = np.dot(np.linalg.inv(P1[:, :3]), points1)
    points2n = np.dot(np.linalg.inv(P2[:, :3]), points2)
    triangulated_points = linear_triangulation(points1n, points2n, P1, P2)
    
    return triangulated_points, (img1, img2, points1, points2, np.random.normal((3,3)), P1, P2)

if __name__ == '__main__':
    image1_idx = 2
    image2_idx = 3
    image1_path = 'images/dino/viff.' + str(image1_idx).zfill(3) + '.ppm'
    image2_path = 'images/dino/viff.' + str(image2_idx).zfill(3) + '.ppm'
    
    projection_matrix1 = scipy.io.loadmat('images/dino/dino_Ps.mat')['P'][0, image1_idx]
    projection_matrix2 = scipy.io.loadmat('images/dino/dino_Ps.mat')['P'][0, image2_idx]
    
    # TODO Check why this is not accurate
    # intrinsics1, _, _ = decompose_projection_matrix(projection_matrix1)
    # intrinsics2, _, _ = decompose_projection_matrix(projection_matrix2)
    # intrinsics1[0, 1] = 0
    # intrinsics2[0, 1] = 0
        
    # Hardcoded intrinsic 
    intrinsics1 = intrinsics2 = np.array([  
        [2360, 0,    360],
        [0,    2360, 288],
        [0,    0,    1]])
        
    np.set_printoptions(suppress=True)
    print(f"Intrinsic matrix 1:\n{intrinsics1}")
    print(f"Intrinsic matrix 2:\n{intrinsics2}")

    triangulated_points, correspondances = reconstruct(image1_path, image2_path, intrinsics1, intrinsics2)
    # triangulated_points, correspondances = reconstruct_with_projection_matrices(image1_path, image2_path, projection_matrix1, projection_matrix2)
    img1, img2, points1, points2, _, _, _ = correspondances
    
    # Display feature correspondences
    fig, ax = plt.subplots(1, 2)
    ax[0].autoscale_view('tight')
    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[0].plot(points1[0], points1[1], 'r.')
    ax[1].autoscale_view('tight')
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[1].plot(points2[0], points2[1], 'r.')
    fig.show()
    
    # Display reconstructed 3D points
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.add_subplot(projection='3d')
    ax.plot(triangulated_points[0], triangulated_points[1], triangulated_points[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    plt.show()    
