# # -*- coding: utf-8 -*-
#
# # Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# # holder of all proprietary rights on this computer program.
# # You can only use this computer program if you have closed
# # a license agreement with MPG or you get the right to use the computer
# # program from someone who is authorized to grant you that right.
# # Any use of the computer program without a valid license is prohibited and
# # liable to prosecution.
# #
# # Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# # der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# # for Intelligent Systems. All rights reserved.
# #
# # Contact: ps-license@tuebingen.mpg.de
#
# import torch
# import joblib
# import numpy as np
# import os.path as osp
# from torch.utils.data import Dataset
#
# # from lib.core.config import VIBE_DB_DIR
# VIBE_DB_DIR = '../VIBE/data/vibe_db'
# # from lib.data_utils.img_utils import split_into_chunks
#
# def split_into_chunks(vid_names, seqlen, stride):
#     video_start_end_indices = []
#
#     video_names, group = np.unique(vid_names, return_index=True)
#     perm = np.argsort(group)
#     video_names, group = video_names[perm], group[perm]
#
#     indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])
#
#     for idx in range(len(video_names)):
#         indexes = indices[idx]
#         if indexes.shape[0] < seqlen:
#             continue
#         chunks = view_as_windows(indexes, (seqlen,), step=stride)
#         start_finish = chunks[:, (0, -1)].tolist()
#         video_start_end_indices += start_finish
#
#     return video_start_end_indices
#
# class AMASS(Dataset):
#     def __init__(self, seqlen):
#         self.seqlen = seqlen
#
#         self.stride = seqlen
#
#         self.db = self.load_db()
#         self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)
#         del self.db['vid_name']
#         print(f'AMASS dataset number of videos: {len(self.vid_indices)}')
#
#     def __len__(self):
#         return len(self.vid_indices)
#
#     def __getitem__(self, index):
#         return self.get_single_item(index)
#
#     def load_db(self):
#         db_file = osp.join(VIBE_DB_DIR, 'amass_db.pt')
#         db = joblib.load(db_file)
#         return db
#
#     def get_single_item(self, index):
#         start_index, end_index = self.vid_indices[index]
#         thetas = self.db['theta'][start_index:end_index+1]
#
#         cam = np.array([1., 0., 0.])[None, ...]
#         cam = np.repeat(cam, thetas.shape[0], axis=0)
#         theta = np.concatenate([cam, thetas], axis=-1)
#
#         target = {
#             'theta': torch.from_numpy(theta).float(),  # cam, pose and shape
#         }
#         return target

import os
import numpy as np
import joblib
from .dataset import Dataset
from src.config import ROT_CONVENTION_TO_ROT_NUMBER
from src import config
from PIL import Image
import sys

sys.path.append('')

# action2motion_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 24, 38]
# change 0 and 8
action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]  # [18,]

from src.utils.action_label_to_idx import action_label_to_idx, idx_to_action_label


def get_z(cam_s, cam_pos, joints, img_size, flength):
    """
    Solves for the depth offset of the model to approx. orth with persp camera.
    """
    # Translate the model itself: Solve the best z that maps to orth_proj points
    joints_orth_target = (cam_s * (joints[:, :2] + cam_pos) + 1) * 0.5 * img_size
    height3d = np.linalg.norm(np.max(joints[:, :2], axis=0) - np.min(joints[:, :2], axis=0))
    height2d = np.linalg.norm(np.max(joints_orth_target, axis=0) - np.min(joints_orth_target, axis=0))
    tz = np.array(flength * (height3d / height2d))
    return float(tz)


def get_trans_from_vibe(vibe, use_z=True):
    alltrans = []
    for t in range(vibe["joints3d"].shape[0]):
        # Convert crop cam to orig cam
        # No need! Because `convert_crop_cam_to_orig_img` from demoutils of vibe
        # does this already for us :)
        # Its format is: [sx, sy, tx, ty]
        cam_orig = vibe["orig_cam"][t]
        x = cam_orig[2]
        y = cam_orig[3]
        if use_z:
            z = get_z(cam_s=cam_orig[0],  # TODO: There are two scales instead of 1.
                      cam_pos=cam_orig[2:4],
                      joints=vibe['joints3d'][t],
                      img_size=480,
                      flength=500)
            # z = 500 / (0.5 * 480 * cam_orig[0])
        else:
            z = 0
        trans = [x, y, z]
        alltrans.append(trans)
    alltrans = np.array(alltrans)
    return alltrans - alltrans[0]


class AMASS(Dataset):
    """
    AMASS Dataset loader for both SMPL-based and G1 retargeted motion data.
    
    Supports two data types:
    1. Standard AMASS: SMPL body model poses (rotation vectors)
    2. G1 Retargeted: Robot DOF positions (joint angles)
    
    Args:
        datapath: Path to dataset file (e.g., 'data/amass/amass_30fps_db.pt')
        split: Dataset split ('train', 'vald', or 'test')
        use_z: Whether to use z translation (default: 1)
        use_g1: If True, loads G1 retargeted data instead of SMPL poses (default: False)
        **kwargs: Additional arguments passed to parent Dataset class
        
    Example:
        # Standard AMASS with SMPL poses
        dataset = AMASS(datapath='./data/amass_db/amass_30fps_db.pt', split='train')
        
        # G1 retargeted data
        dataset = AMASS(datapath='./data/g1_amass_db/g1_amass_30fps_db.pt', 
                       split='train', use_g1=True, pose_rep='dof')
    """
    dataname = "amass"

    def __init__(self, datapath="data/amass/amass_30fps_legacy_db.pt", split="train", use_z=1, **kwargs):
        assert '_db.pt' in datapath
        self.datapath = datapath.replace('_db.pt', '_{}.pt'.format(split))
        assert os.path.exists(self.datapath)
        print('datapath used by amass is [{}]'.format(self.datapath))
        super().__init__(**kwargs)

        self.dataname = "amass"
        self.use_g1 = kwargs.get('use_g1', False)

        # FIXME - hardcoded:
        self.rot_convention = 'legacy' if not self.use_g1 else 'g1_dof'
        self.use_betas = False
        self.use_gender = False
        self.use_body_features = False
        if 'clip_preprocess' in kwargs.keys():
            self.clip_preprocess = kwargs['clip_preprocess']

        self.use_z = (use_z != 0)

        # keep_actions = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
        dummy_class = [0]
        genders = config.GENDERS
        self.num_classes = len(dummy_class)

        self.db = self.load_db()
        self._joints3d = []
        self._poses = []
        self._num_frames_in_video = []
        self._actions = []
        self._betas = []
        self._genders = []
        self._heights = []
        self._masses = []
        self._clip_images = []
        self._clip_texts = []
        self._clip_pathes = []
        self._actions_cat = []
        # PROBLEM: Which is better?
        self.clip_label_text = "text_raw_labels"  # "text_proc_labels"

        seq_len = 100
        n_sequences = len(self.db['thetas'])
        # split sequences
        for seq_idx in range(n_sequences):
            n_sub_seq = self.db['thetas'][seq_idx].shape[0] // seq_len
            if n_sub_seq == 0: continue
            n_frames_in_use = n_sub_seq * seq_len
            joints3d = np.split(self.db['joints3d'][seq_idx][:n_frames_in_use], n_sub_seq)
            poses = np.split(self.db['thetas'][seq_idx][:n_frames_in_use], n_sub_seq)
            self._joints3d.extend(joints3d)
            self._poses.extend(poses)
            self._num_frames_in_video.extend([seq_len] * n_sub_seq)

            if 'action_cat' in self.db:
                self._actions_cat.extend(np.split(self.db['action_cat'][seq_idx][:n_frames_in_use], n_sub_seq))

            if self.use_betas:
                self._betas.extend(np.split(self.db['betas'][seq_idx][:n_frames_in_use], n_sub_seq))
            if self.use_gender:
                self._genders.extend([str(self.db['genders'][seq_idx]).replace("b'female'", "female").replace("b'male'",
                                                                                                              "male")] * n_sub_seq)
            if self.use_body_features:
                self._heights.extend([self.db['heights'][seq_idx]] * n_sub_seq)
                self._masses.extend([self.db['masses'][seq_idx]] * n_sub_seq)
            if 'clip_images' in self.db.keys():
                images = [np.squeeze(e) for e in np.split(self.db['clip_images'][seq_idx][:n_sub_seq], n_sub_seq)]
                processed_images = [self.clip_preprocess(Image.fromarray(img)) for img in images]
                self._clip_images.extend(processed_images)
            if self.clip_label_text in self.db:
                self._clip_texts.extend(np.split(self.db[self.clip_label_text][seq_idx][:n_frames_in_use], n_sub_seq))
            if 'clip_pathes' in self.db:
                self._clip_pathes.extend(np.split(self.db['clip_pathes'][seq_idx][:n_sub_seq], n_sub_seq))
            if 'clip_images_emb' in self.db.keys():
                self._clip_images_emb.extend(np.split(self.db['clip_images_emb'][seq_idx][:n_sub_seq], n_sub_seq))



            actions = [0] * n_sub_seq
            self._actions.extend(actions)

        assert len(self._num_frames_in_video) == len(self._poses) == len(self._joints3d) == len(self._actions)
        if self.use_betas:
            assert len(self._poses) == len(self._betas)
        if self.use_gender:
            assert len(self._poses) == len(self._genders)
        if 'clip_images' in self.db.keys():
            assert len(self._poses) == len(self._clip_images)

        self._actions = np.array(self._actions)
        self._num_frames_in_video = np.array(self._num_frames_in_video)

        N = len(self._poses)
        # same set for training and testing
        self._train = np.arange(N)
        self._test = np.arange(N)

        self._action_to_label = {x: i for i, x in enumerate(dummy_class)}
        self._label_to_action = {i: x for i, x in enumerate(dummy_class)}

        self._gender_to_label = {x: i for i, x in enumerate(genders)}
        self._label_to_gender = {i: x for i, x in enumerate(genders)}

        self._action_classes = idx_to_action_label

    def load_db(self):
        # Load amass dataset encoded to a .db file
        # The loaded data is structured:
        # {
        #     'theta': [data_size, 82] (float64) (structured [pose(72), betas(10)])
        #     'vid_name': [data_size] (str)
        # }
        # data_size should be [16275369]
        db_file = self.datapath
        db = joblib.load(db_file)

        if 'clip_images' in db and db['clip_images'][0] is None:  # No images added
            del db['clip_images']

        return db

    def _load_joints3D(self, ind, frame_ix):
        joints3D = self._joints3d[ind][frame_ix]
        return joints3D

    def _load_dof_positions(self, ind, frame_ix):
        """Load DOF positions for G1 retargeted data."""
        dof_pos = self._poses[ind][frame_ix, :]  # [seq_len, num_dofs]
        return dof_pos

    def _load_rotvec(self, ind, frame_ix):
        if self.use_g1:
            # For G1, return DOF positions directly without reshaping
            return self._poses[ind][frame_ix, :]
        pose = self._poses[ind][frame_ix, :].reshape(-1, ROT_CONVENTION_TO_ROT_NUMBER[self.rot_convention] + 1,
                                                     3)  # +1 for global orientation
        return pose

    def _load_betas(self, ind, frame_ix):
        betas = self._betas[ind][frame_ix].transpose((1, 0))
        return betas

    def _load_gender(self, ind, frame_ix):
        gender = self._gender_to_label[self._genders[ind]]
        return gender

    def _load_body_features(self, ind, frame_ix):
        return {'mass': float(self._masses[ind]), 'height': float(self._heights[ind])}


class G1AMASS(Dataset):
    """
    G1 Retargeted AMASS Dataset loader for robot motion data.
    
    Loads G1 retargeted AMASS data with:
    - body_positions: (T, 30, 3) - 3D positions of robot bodies (primary motion representation)
    - dof_positions: (T, 29) - joint angles in radians
    - dof_velocities: (T, 29) - joint velocities
    - body_rotations: (T, 30, 4) - quaternion orientations
    - body_linear_velocities: (T, 30, 3)
    - body_angular_velocities: (T, 30, 3)
    
    Args:
        datapath: Path to G1 dataset file (e.g., 'data/g1_amass_db/g1_amass_30fps_train.pt')
        split: Dataset split ('train', 'vald', or 'test')
        **kwargs: Additional arguments passed to parent Dataset class
        
    Example:
        dataset = G1AMASS(datapath='./data/g1_amass_db/g1_amass_30fps_db.pt', 
                         split='train', pose_rep='xyz')
    """
    dataname = "g1_amass"

    def __init__(self, datapath="data/g1_amass_db/g1_amass_30fps_db.pt", split="train", **kwargs):
        assert '_db.pt' in datapath
        self.datapath = datapath.replace('_db.pt', '_{}.pt'.format(split))
        assert os.path.exists(self.datapath), f"G1 AMASS dataset not found: {self.datapath}"
        print('G1 AMASS datapath: [{}]'.format(self.datapath))
        super().__init__(**kwargs)

        self.dataname = "g1_amass"
        
        # G1 specific settings
        self.rot_convention = 'g1_xyz'  # Using body positions (xyz)
        self.use_betas = False  # G1 has fixed geometry
        self.use_gender = False  # Not applicable for robot
        self.use_body_features = False
        
        if 'clip_preprocess' in kwargs.keys():
            self.clip_preprocess = kwargs['clip_preprocess']

        dummy_class = [0]
        self.num_classes = len(dummy_class)

        self.db = self.load_db()
        
        # G1 specific data storage
        self._body_positions = []      # (T, 30, 3) - primary motion representation
        self._dof_positions = []       # (T, 29) - joint angles
        self._dof_velocities = []      # (T, 29) - joint velocities
        self._body_rotations = []      # (T, 30, 4) - quaternions
        self._body_linear_velocities = []   # (T, 30, 3)
        self._body_angular_velocities = []  # (T, 30, 3)
        
        # Common data storage (same as AMASS)
        self._poses = []  # Will store body_positions reshaped to (T, 30*3)
        self._num_frames_in_video = []
        self._actions = []
        self._clip_images = []
        self._clip_texts = []
        self._clip_pathes = []
        self._actions_cat = []
        
        self.clip_label_text = "text_raw_labels"

        seq_len = 100
        n_sequences = len(self.db['body_positions'])
        
        # Split sequences
        for seq_idx in range(n_sequences):
            # The list of All motions of all Sequences & subjects
            body_pos = self.db['body_positions'][seq_idx]
            n_sub_seq = body_pos.shape[0] // seq_len
            if n_sub_seq == 0:
                continue
            n_frames_in_use = n_sub_seq * seq_len
            
            # Split body positions (primary data)
            body_positions = np.split(body_pos[:n_frames_in_use], n_sub_seq)
            self._body_positions.extend(body_positions)
            
            # Reshape body positions to (T, 90) for compatibility with pose representation
            poses_flat = [bp.reshape(seq_len, -1) for bp in body_positions]
            self._poses.extend(poses_flat)
            
            # Split DOF positions
            if 'dof_positions' in self.db:
                dof_pos = np.split(self.db['dof_positions'][seq_idx][:n_frames_in_use], n_sub_seq)
                self._dof_positions.extend(dof_pos)
            
            # Split DOF velocities
            if 'dof_velocities' in self.db:
                dof_vel = np.split(self.db['dof_velocities'][seq_idx][:n_frames_in_use], n_sub_seq)
                self._dof_velocities.extend(dof_vel)
            
            # Split body rotations
            if 'body_rotations' in self.db:
                body_rot = np.split(self.db['body_rotations'][seq_idx][:n_frames_in_use], n_sub_seq)
                self._body_rotations.extend(body_rot)
            
            # Split body linear velocities
            if 'body_linear_velocities' in self.db:
                body_lin_vel = np.split(self.db['body_linear_velocities'][seq_idx][:n_frames_in_use], n_sub_seq)
                self._body_linear_velocities.extend(body_lin_vel)
            
            # Split body angular velocities
            if 'body_angular_velocities' in self.db:
                body_ang_vel = np.split(self.db['body_angular_velocities'][seq_idx][:n_frames_in_use], n_sub_seq)
                self._body_angular_velocities.extend(body_ang_vel)
            
            self._num_frames_in_video.extend([seq_len] * n_sub_seq)
            
            # Action categories
            if 'action_cat' in self.db:
                self._actions_cat.extend(np.split(self.db['action_cat'][seq_idx][:n_frames_in_use], n_sub_seq))
            
            # CLIP data
            if 'clip_images' in self.db.keys() and self.db['clip_images'][seq_idx] is not None:
                images = [np.squeeze(e) for e in np.split(self.db['clip_images'][seq_idx][:n_sub_seq], n_sub_seq)]
                processed_images = [self.clip_preprocess(Image.fromarray(img)) for img in images]
                self._clip_images.extend(processed_images)
            
            if self.clip_label_text in self.db:
                self._clip_texts.extend(np.split(self.db[self.clip_label_text][seq_idx][:n_frames_in_use], n_sub_seq))
            
            if 'clip_pathes' in self.db:
                self._clip_pathes.extend(np.split(self.db['clip_pathes'][seq_idx][:n_sub_seq], n_sub_seq))
            
            # Dummy actions
            actions = [0] * n_sub_seq
            self._actions.extend(actions)

        assert len(self._num_frames_in_video) == len(self._poses) == len(self._body_positions) == len(self._actions)
        
        self._actions = np.array(self._actions)
        self._num_frames_in_video = np.array(self._num_frames_in_video)

        N = len(self._poses)
        self._train = np.arange(N)
        self._test = np.arange(N)

        self._action_to_label = {x: i for i, x in enumerate(dummy_class)}
        self._label_to_action = {i: x for i, x in enumerate(dummy_class)}

        self._action_classes = idx_to_action_label
        
        print(f'G1AMASS dataset loaded: {N} sequences, {sum(self._num_frames_in_video)} frames total')

    def load_db(self):
        """Load G1 retargeted AMASS dataset from .pt file."""
        db_file = self.datapath
        db = joblib.load(db_file)

        if 'clip_images' in db and db['clip_images'][0] is None:
            del db['clip_images']

        return db

    # Legacy Support
    def _load_joints3D(self, ind, frame_ix):
        """Load 3D body positions (equivalent to joints3D for G1)."""
        body_pos = self._body_positions[ind][frame_ix]  # (30, 3)
        return body_pos

    def _load_rotvec(self, ind, frame_ix):
        """
        Load rotation vectors for G1.
        For G1, we return body_positions reshaped to match expected format.
        Shape: (seq_len, 30, 3)
        """
        body_pos = self._body_positions[ind][frame_ix]  # (30, 3)
        # Reshape to (30, 1, 3) to match expected format (num_bodies, 1, 3)
        return body_pos.reshape(30, 1, 3)

    # New G1-specific loaders
    def _load_dof_positions(self, ind, frame_ix):
        """Load DOF positions for G1 robot."""
        if len(self._dof_positions) > 0:
            return self._dof_positions[ind][frame_ix]
        return None

    def _load_dof_velocities(self, ind, frame_ix):
        """Load DOF velocities for G1 robot."""
        if len(self._dof_velocities) > 0:
            return self._dof_velocities[ind][frame_ix]
        return None

    def _load_body_positions(self, ind, frame_ix):
        """Load 3D body positions (equivalent to joints3D for G1)."""
        if len(self._dof_positions) > 0:
            return self._body_positions[ind][frame_ix]  # (30, 3)
        return None

    def _load_body_rotations(self, ind, frame_ix):
        """Load body rotations (quaternions) for G1 robot."""
        if len(self._body_rotations) > 0:
            return self._body_rotations[ind][frame_ix]
        return None


if __name__ == "__main__":
    dataset = AMASS()
    # Test G1AMASS
    # g1_dataset = G1AMASS(datapath='./data/g1_amass_db/g1_amass_30fps_db.pt', split='train')
