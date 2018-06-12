import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
import sys

def define_actions( action ):
  """
  Given an action string, returns a list of corresponding actions.

  Args
    action: String. either "all" or one of the h36m actions
  Returns
    actions: List of strings. Actions to use.
  Raises
    ValueError: if the action is not a valid action in Human 3.6M
  """
  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]
  if action == "All" or action == "all":
    return actions
  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )
  return [action]


def to_image_frame(pose2ds, pose3ds):
    """
    Align 3D pose with the 2D pose; subject to orthogonal projection
    pose2ds: n * 14 * 2
    pose3ds: n * 14 * 3
    """
    n = pose2ds.shape[0]
    dim = pose2ds.shape[1]
    ifposes = []
    for i in range(n):
        pose2d = pose2ds[i]
        pose3d = pose3ds[i]
        extend_pose2d = np.concatenate([pose2d, np.zeros((dim, 1))], axis=1)
        _, res, _ = procrustes(extend_pose2d, pose3d)
        ifposes.append(res)
    return np.array(ifposes)


def procrustes(A, B, scaling=True, reflection='best'):
    assert A.shape[0] == B.shape[0]
    n, dim_x = A.shape
    _, dim_y = B.shape

    # remove translation
    A_bar = A.mean(0)
    B_bar = B.mean(0)
    A0 = A - A_bar
    B0 = B - B_bar

    # remove scale
    ssX = (A0**2).sum()
    ssY = (B0**2).sum()
    A_norm = np.sqrt(ssX)
    B_norm = np.sqrt(ssY)
    A0 /= A_norm
    B0 /= B_norm

    if dim_y < dim_x:
        B0 = np.concatenate((B0, np.zeros(n, dim_x - dim_y)), 0)

    # optimum rotation matrix of B
    A = np.dot(A0.T, B0)
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T
    R = np.dot(V, U.T)

    if reflection is not 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(R) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            R = np.dot(V, U.T)

    S_trace = s.sum()
    if scaling:
        # optimum scaling of B
        scale = S_trace * A_norm / B_norm

        # standarised distance between A and scale*B*R + c
        d = 1 - S_trace**2

        # transformed coords
        Z = A_norm * S_trace * np.dot(B0, R) + A_bar
    else:
        scale = 1
        d = 1 + ssY / ssX - 2 * S_trace * B_norm / A_norm
        Z = B_norm * np.dot(B0, R) + A_bar

    # transformation matrix
    if dim_y < dim_x:
        R = R[:dim_y, :]
    translation = A_bar - scale * np.dot(B_bar, R)

    # transformation values
    tform = {'rotation': R, 'scale': scale, 'translation': translation}
    return d, Z, tform