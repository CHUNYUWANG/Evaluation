import numpy as np
from params import params
import load_data as ld
import pose_util as pu
import sys


def create_testset(pose2d, pose3d, params):
    """
    Input:
    pose2d: groundtruth 2D poses in H3.6M (n, 14, 2)
    pose3d: groundtruth 3D poses in H3.6M (n, 14, 3)

    Return:
    cpose2d: corrupted 2D poses (m, 14, 2)
    cpose3d: groundtruth 3D poses corresponding to the cpose2d (m, 14, 3)
    cmask  : 0 -> not corrupted; 1 -> corrupted (m, 14, 2)
    """
    pose2d_gauss, pose3d_gauss, masks_gauss = add_gaussian_noise(pose2d.copy(), pose3d.copy(), params)
    pose2d_miss, pose3d_miss, masks_miss = add_missing_noise(pose2d.copy(), pose3d.copy(), params)
    pose2d_tran, pose3d_tran, masks_tran = add_transformation_noise(pose2d, pose3d, params)
    return ( np.concatenate((pose2d_gauss, pose2d_miss, pose2d_tran), axis=0),
            np.concatenate((pose3d_gauss, pose3d_miss, pose3d_tran), axis=0),
            np.concatenate((masks_gauss, masks_miss, masks_tran), axis=0))

def add_gaussian_noise(pose2d, pose3d, params):
    means = params.gaussian.means
    variances = params.gaussian.vars
    percent = params.gaussian.percent
    max_joint_num = params.gaussian.max_joint_num
    num, njoints, dim = pose2d.shape
    num_split = [int(np.floor(num*perc)) for perc in percent]
    num_split[-1] = num - sum(num_split[:-1])

    noises = []
    masks_all = []
    for mean, var, ns in zip(means, variances, num_split):
        noise = np.random.normal(loc=mean, scale=var, size=(ns, njoints, dim))
        zeros = np.zeros(shape=(ns, njoints, dim))
        masks = []
        for _ in range(ns):
            c = np.random.randint(0, max_joint_num)
            mask = np.array([1]*c + [0]*(njoints-c))
            np.random.shuffle(mask)
            masks.append(np.tile(np.reshape(mask, (njoints, 1)), (1, dim)))
        masks = np.array(masks)
        noise = noise*masks+zeros*(1-masks)
        noises.append(noise)
        masks_all.append(masks)
    noises = np.concatenate(noises, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)
    return pose2d+noises, pu.to_image_frame(pose2d, pose3d), masks_all



def add_missing_noise(pose2d, pose3d, params):
    """
    Note: using Guoqiang's strategy
    :return
    pose2d: processed pose2d with missing joints being set to 0
    pose3d:
    mask_all: same shape as pose2d, 1 indicates missing joints and 0 for others
    """
    pose2d_origin = pose2d.copy()
    num = pose2d.shape[0]
    mask_all = []
    for p in range(num):
        rand_x = np.random.randint(0, params.image.width - 1)
        rand_y = np.random.randint(0, params.image.height - 1)

        rand_num = np.random.randint(0, params.miss.max_joint_num)

        dis_joint = np.sum((pose2d[p] - np.array([rand_x, rand_y])) ** 2, axis=1)  # 14,
        mask_idx = np.argsort(dis_joint)[:rand_num]  # index of marked joints

        mask = np.zeros_like(pose2d[p])
        mask[mask_idx, :] = 1.0
        pose2d[p] *= (1.0 - mask)
        mask_all.append(mask)
    mask_all = np.stack(mask_all)
    return pose2d, pu.to_image_frame(pose2d_origin, pose3d), mask_all


def add_transformation_noise(pose2d, pose3d, params):
    nposes = pose2d.shape[0]
    scale_range = params.transform.scale_range
    translate_range = params.transform.translate_range
    rotate_range = params.transform.rotate_range
    pose_transformed = []
    for i in range(nposes):
        pose_src = pose2d[i].copy()
        # (1) scaling
        scale = np.random.uniform(scale_range[0], scale_range[1])
        pose_src *= scale
        # (2) translate
        offset_x = np.random.uniform(translate_range[0], translate_range[1])
        offset_y = np.random.uniform(translate_range[0], translate_range[1])
        pose_src += np.array([[offset_x], [offset_y]]).T
        # (3) rotation
        rotdeg = np.random.uniform(rotate_range[0], rotate_range[1])
        rotdeg = 0
        rotmat = np.array(
            [[np.cos(rotdeg), -np.sin(rotdeg)],
             [np.sin(rotdeg), np.cos(rotdeg)]]
        )
        pose_src = np.dot(rotmat, pose_src.T).T
        # (3) rotation
        pose_transformed.append(pose_src)
    pose_transformed = np.array(pose_transformed)
    masks = np.zeros_like(pose_transformed)
    return pose_transformed, pu.to_image_frame(pose2d, pose3d), masks 

if __name__=='__main__':
    actions = pu.define_actions('all')
    pose2ds, pose3ds = ld.load_data(actions, is_train=False)

    pose2ds_complete = []
    pose3ds_complete = []
    for key in pose2ds.keys():
        pose2ds_complete.append(pose2ds[key])
        pose3ds_complete.append(pose3ds[key])
    pose2ds_complete = np.concatenate(pose2ds_complete, axis=0)
    pose2ds_complete = np.reshape(pose2ds_complete, (-1,14,2))
    pose3ds_complete = np.concatenate(pose3ds_complete, axis=0)
    pose3ds_complete = np.reshape(pose3ds_complete, (-1,14,3))

    cpose2d, cpose3d, cmasks = create_testset(pose2ds_complete, pose3ds_complete, params)









