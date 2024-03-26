import numpy as np

def find_null_space(A):
    '''
    Given matrix A, find its null space x such that Ax = 0
    '''
    _, _, Vt = np.linalg.svd(A)
    return Vt[-1]

def normalize_rotation_matrix(R):
    '''
    Given rotation matrix R, normalize it such that its determinant is 1
    '''
    assert R.shape == (3, 3)
    U, _, Vt = np.linalg.svd(R)
    
    if np.linalg.det(np.dot(U, Vt)) < 0:
        Vt[-1] *= -1
    return np.dot(U, Vt)
    
def rq_decomposition(M):
    '''
    Given matrix M, perform RQ decomposition such that M = RQ
    '''
    Q, R = np.linalg.qr(np.flipud(M).T)
    R = np.flipud(R.T)
    return R[:, ::-1], Q.T[::-1, :]

# TODO Verify the correctness of this function 
def decompose_projection_matrix(P):
    '''
    Given projection matrix P, decompose it into K, R, t such that P = K[R|t]
    
    @return K: 3x3 camera calibration matrix
    @return R: 3x3 rotation matrix
    @return t: 3x1 translation vector
    '''
    K, R = rq_decomposition(P[:, :3])
    
    # make diagonal of K positive
    T = np.diag(np.sign(np.diag(K)))
    if np.linalg.det(T) < 0:
        T[1, 1] *= -1

    K = np.dot(K, T)
    R = np.dot(T, R)  # T is its own inverse
    t = np.dot(np.linalg.inv(np.dot(K, R)), P[:, 3])
    
    K = K/K[2, 2]
    R = normalize_rotation_matrix(R)

    return K, R, t
