from sklearn.preprocessing import normalize
import numpy as np

from params import params

def pose_preprocess(joints_3d):
    """Transform pose from world coordinate system into object coordinate system.

    Args:
        joints_3d: A [n_total, n_joint, 3] pose matrix with world coordinate
            system.

    Returns:
        A [n_total, n_joint, 3] pose matrix with object coordinate system.
    """
    idx_right_hip = params.joint_config.idx_right_hip
    idx_left_hip = params.joint_config.idx_left_hip
    idx_neck = params.joint_config.idx_neck
    joints_3d = scale_normalize(joints_3d)
    x_axis = joints_3d[:, idx_right_hip] - joints_3d[:, idx_left_hip]
    x_axis = normalize(x_axis, axis=1)

    mid_hip = 0.5 * (joints_3d[:, idx_right_hip] + joints_3d[:, idx_left_hip])
    y_axis = joints_3d[:, idx_neck] - mid_hip
    y_axis = normalize(y_axis, axis=1)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = normalize(z_axis, axis=1)

    project_matrixs = np.stack([x_axis, y_axis, z_axis, mid_hip], axis=-1)
    res = [
        _apply_rotate(project, pose)
        for project, pose in zip(project_matrixs, joints_3d)
    ]
    return np.stack(res)

def _apply_rotate(project, pose):
    """Apply projection.

    Args:
        project: [3, 4] matrix, implies rotation and translation.
        pose: [n_joint, 3] matrix.

    Returns:
        Projected pose, a [n_joint, 3] matrix.
    """
    bottom = np.array([0, 0, 0, 1]).reshape([1, 4]).astype(np.double)
    ones = np.ones([1, pose.shape[0]])

    project = np.concatenate([project, bottom])
    project_inverse = np.linalg.inv(project)
    x_expand = np.concatenate([pose.T, ones], axis=0)
    new_pose = np.matmul(project_inverse, x_expand)[:3].T
    return new_pose



def scale_normalize(joints):
    """Normalize pose with knee-neck height 9.2 dm.

    Args:
        joints: [n_total, n_joint, 3] pose array.

    Returns:
        A normalized [n_total, n_joint, 3] pose array
    """
    idx_left_hip = params.joint_config.idx_left_hip
    idx_left_knee = params.joint_config.idx_left_knee
    n_total = joints.shape[0]
    joints_norm = np.zeros_like(joints)
    for i in range(n_total):
        j3d = joints[i]
        knee_neck = np.linalg.norm(j3d[idx_left_hip] - j3d[idx_left_knee])
        knee_neck = knee_neck if knee_neck != 0 else 1e-5
        j3d_norm = j3d / knee_neck
        joints_norm[i] = j3d_norm
    return joints_norm