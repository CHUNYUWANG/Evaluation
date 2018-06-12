#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import numpy as np

# ###################
# joints
# 0 - 'RHip'
# 1 - 'RKnee'
# 2 - 'RFoot'
# 3 - 'LHip'
# 4 - 'LKnee'
# 5 - 'LFoot'
# 6 - 'Thorax'
# 7 - 'Head'
# 8 - 'LShoulder'
# 9 - 'LElbow'
# 10 - 'LWrist'
# 11 - 'RShoulder'
# 12 - 'RElbow'
# 13 - 'RWrist'
# ##################


def load_data(actions, path='./data', is_train=True):
    # actions: list of actions, e.g., [Walking, WalkingDog]
    # path: path to data

    pose2d, pose3d = {}, {}
    for act in actions:
        pose2d.update({act: []})
        pose3d.update({act: []})

    file_2d = os.path.join(path, 'train' + '_2d.json') if is_train else os.path.join(path, 'test' + '_2d.json')
    file_3d = os.path.join(path, 'train' + '_3d.json') if is_train else os.path.join(path, 'test' + '_3d.json')

    with open(file_2d, 'r') as f:
        pose_2d = json.load(f)
    with open(file_3d, 'r') as f:
        pose_3d = json.load(f)

    for k in pose_2d.keys():
        act = k.split('.')[1].split(' ')[0]
        if act in actions:
            for i in pose_2d[k]:
                pose2d[act].append(i)
            for j in pose_3d[k]:
                pose3d[act].append(j)

    for k in pose2d.keys():
        pose2d.update({k: np.array(pose2d[k])})
        pose3d.update({k: np.array(pose3d[k])})

    # pose2d/pose3d: dict of pose:
    # key: actions, e.g., 'Walking'
    # value: pose, e.g., Nx 24 for pose2d['Walking']
    return pose2d, pose3d


