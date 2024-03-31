import numpy as np

def euclidean_to_homogeneous(points):
    '''
    Converts Nx3 or Nx2 Euclidean coordinates to Nx4 or Nx3 Homogeneous coordinates.
    '''
    assert points.shape[1] == 3 or points.shape[1] == 2
    return np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

def homogeneous_to_euclidean(points):
    '''
    Converts Nx4 or Nx3 Homogeneous coordinates to Nx3 or Nx2 Euclidean coordinates.
    '''
    assert points.shape[1] == 4 or points.shape[1] == 3
    return points[:, :-1] / points[:, -1:]

def transformation_matrix(R, t):
    '''
    Given rotation matrix R and translation vector t, calculate the transformation matrix T
    T = | R t |
        | 0 1 |
    '''
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    return np.concatenate([np.concatenate([R, t.reshape(-1, 1)], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

def extract_rotation_translation(T):
    '''
    Given transformation matrix T, extract rotation matrix R and translation vector t
    '''
    assert T.shape == (4, 4)
    return T[:3, :3], T[:3, 3]
    
def transform_points(points, T):
    '''
    Given Nx3 Euclidean points and transformation matrix T, transform the points
    '''
    assert points.shape[1] == 3
    assert T.shape == (4, 4)
    return homogeneous_to_euclidean(np.dot(euclidean_to_homogeneous(points), T.T))

def invert_transformation(T):
    '''
    Given transformation matrix T, calculate its inverse
    '''
    assert T.shape == (4, 4)
    R, t = extract_rotation_translation(T)
    R_inv = R.T
    t_inv = -np.dot(R_inv, t)
    return transformation_matrix(R_inv, t_inv)
    