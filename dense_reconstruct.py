import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt

import pangolin 
from OpenGL.GL import glEnable, glClear, glClearColor, glPointSize, glBegin, glEnd, glColor3f, glVertex3f, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_POINTS
from OpenGL.GL import GL_DEPTH_TEST

from reconstruct import reconstruct
from utils.mvg import calculate_epipolar_lines, block_matching, linear_triangulation
from utils.linalg import decompose_projection_matrix
from utils.transformations import euclidean_to_homogeneous

def display_3d_points(points, image, pixels):
    pangolin.CreateWindowAndBind('3D Points Display', 640, 480)
    glEnable(GL_DEPTH_TEST)
    
    # Define projection and initial camera pose
    projection = pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.01, 100)
    view = pangolin.ModelViewLookAt(0, 0, -2, 0, 0, 0, pangolin.AxisDirection.AxisY)
    scam = pangolin.OpenGlRenderState(projection, view)
    handler = pangolin.Handler3D(scam)
    
    viewport = pangolin.CreateDisplay()
    viewport.SetBounds(0.0, 1.0, 0.0, 1.0, -640/480)
    viewport.SetHandler(handler)
    
    while not pangolin.ShouldQuit():
        # Clear screen and activate view
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        viewport.Activate(scam)
        
        # Render 3D points
        glPointSize(2.0)  
        glBegin(GL_POINTS)
        for i in range(points.shape[0]):
            point = points[i]
            pixel = pixels[i]
            color = image[int(pixel[1]), int(pixel[0])]/255
            glColor3f(color[0], color[1], color[2])
            glVertex3f(point[0], point[1], point[2])
        glEnd()        
        pangolin.FinishFrame()
        
def dense_reconstruct(img1, img2, E, P1, P2, intrinsics, points1, stride=50):
    # Get all pixel coordinates
    # y_range = np.arange(0, img1.shape[0], stride)
    # x_range = np.arange(0, img1.shape[1], stride)
    # yy, xx = np.meshgrid(y_range, x_range)
    # pixel_coords = np.vstack([xx.flatten(), yy.flatten()])
    # pixel_coords = euclidean_to_homogeneous(pixel_coords.T)
    pixel_coords = points1.T
    
    # Calculate epipolar lines
    lines = calculate_epipolar_lines(pixel_coords.T, E, intrinsics)
    # Note: This is not going to be one to one
    triangulated_points = [] 
    pixel_points = []
    for i in range(pixel_coords.shape[0]):
        print(f"Processing pixel {i}/{pixel_coords.shape[0]}")
        matching_pix = block_matching(pixel_coords[i, :2], lines[:, i], img1, img2)
        if matching_pix is None:
            print(f"No matching pixel found for pixel {i}")
            continue
        matching_pix = np.array([[matching_pix[0]], [matching_pix[1]], [1]])
        pointn1 = np.dot(np.linalg.inv(intrinsics), pixel_coords[i, :]).reshape(-1, 1)
        pointn2 = np.dot(np.linalg.inv(intrinsics), matching_pix)
        triangulated_point = linear_triangulation(pointn1, pointn2, P1, P2)
        triangulated_points.append(triangulated_point)
        pixel_points.append(pixel_coords[i, :2].astype(int))
    
    return np.array(triangulated_points), np.array(pixel_points)
        
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
    img1, img2, points1, points2, E, P1, P2 = correspondances
    
    # index = 200
    # stride = 5
    # lines = calculate_epipolar_lines(points1, E, intrinsics)    
    # matching_pix = block_matching(points1[:2, index], lines[:, index], img1, img2)    
    
    # # Image 1
    # fig = plt.figure()    
    # fig.suptitle('Pixel in Image 1', fontsize=16)
    # plt.imshow(img1)
    # plt.scatter(points1[0][index], points2[1][index], color='red', s=50)  # Draw a red point
        
    # # Image 2
    # fig = plt.figure()
    # fig.suptitle('Epipolar Line in Image 2', fontsize=16)
    # plt.imshow(img2)
    # rows, cols, _ = img2.shape
    # y_vals = np.arange(rows)
    # x_vals = (-lines[1][index] * y_vals - lines[2][index]) / lines[0][index]
    # valid_indices = np.logical_and(x_vals >= 0, x_vals < cols) 
    # plt.plot(x_vals[valid_indices], y_vals[valid_indices], color='blue')
    # plt.scatter(matching_pix[0], matching_pix[1], color='red', s=50)  # Draw a red point        
    # plt.show()
    
    dense_triangulated_points, pixel_points = dense_reconstruct(img1, img2, E, P1, P2, intrinsics, points1)
    display_3d_points(dense_triangulated_points, img1, pixel_points)
    