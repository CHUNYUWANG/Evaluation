"""
Parameter files to create testing dataset
"""

from easydict import EasyDict as edict
import numpy as np

params = edict()

# Gaussian noise parameters
params.gaussian = edict()
means = [0, 0, 0, 0, 0, 0]
variance = [1, 2, 3, 4, 5, 6]
percent = [1.0/len(means)]*len(means) # number of corrupted poses in each configuration zip(means, variance, percent)
params.gaussian.means = means
params.gaussian.vars = variance
params.gaussian.percent = percent 
params.gaussian.max_joint_num = 5

# Transformation parameters: scale, translate, rotate
params.transform = edict()
params.transform.scale_range = [0.1, 10]
params.transform.translate_range = [-300, 300]
params.transform.rotate_range = [0, 2*np.pi]


# Joint configurations
params.joint_config = edict()
params.joint_config.idx_right_hip = 2
params.joint_config.idx_left_hip = 3
params.joint_config.idx_left_knee = 4
params.joint_config.idx_neck = 6

# Image size parameters
params.image = edict()
params.image.height = 1000
params.image.width = 1000

# Shadow box parameters
params.box = edict()
params.box.max_width = 100
params.box.min_width = 10
params.box.max_height = 100
params.box.min_height = 10

params.miss = edict()
params.miss.max_joint_num = 5


