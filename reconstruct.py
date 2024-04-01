import numpy as np
import scipy
from matplotlib import pyplot as plt
import cv2

from utils.mvg import find_sift_correspondences, compute_fundamental_or_essential_matrix, \
    compute_P2_from_P1, linear_triangulation
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
    
def reconstruct(img1_path, img2_path, intrinsics):
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
    
    points1n = np.dot(np.linalg.inv(intrinsics), points1)
    points2n = np.dot(np.linalg.inv(intrinsics), points2)
    E = compute_fundamental_or_essential_matrix(points1n, points2n, compute_essential=True)
    
    # Assuming the first camera is at the origin
    P1 = np.eye(3, 4)    
    P2 = compute_P2_from_P1(E, P1, points1n[:, 0], points2n[:, 0])

    triangulated_points = linear_triangulation(points1n, points2n, P1, P2)    
    return triangulated_points, (img1, img2, points1, points2, E, P1, P2)


if __name__ == '__main__':
    image1_path = 'images/dino/viff.000.ppm'
    image2_path = 'images/dino/viff.001.ppm'
    projection_matrix = scipy.io.loadmat('images/dino/dino_Ps.mat')['P'][0, 0]
    
    # TODO Check why this is not accurate
    intrinsics, _, _ = decompose_projection_matrix(projection_matrix)
    
    # Hardcoded intrinsic 
    intrinsics = np.array([  
        [2360, 0,    360],
        [0,    2360, 288],
        [0,    0,    1]])
        
    triangulated_points, correspondances = reconstruct('images/dino/viff.000.ppm', 'images/dino/viff.001.ppm', intrinsics)
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
