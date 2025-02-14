import torch
from torch.utils.data import Dataset, DataLoader
import os
from PAR.datamodules.components.dexycb_utils import *
import trimesh
from PAR.datamodules.components.get_data_splits import get_split
from scipy.spatial.transform import Rotation as R
import numpy as np
import yaml
import cv2

MIN_TX = np.array([-0.13972618, -0.3158688, 0.49080718])
MAX_TX = np.array([0.4228604, 0.2755164, 1.0603081])
HOME_DIR = os.environ['HOME']

class DexYCB(Dataset):
    def __init__(self, dexycb_root=os.path.join(HOME_DIR, 'datasets/dexycb'), split='train', 
                relative_norm=False, rel_per_frame=False):
        '''
        Args:
            dexycb_root (string): Directory with all the data.
            split (string): 'train', 'val', or 'test' split of the dataset
            relative_norm (bool): Normalize the hand joints and object poses relative to the hand position
            rel_per_frame (bool): Normalize the hand joints and object poses relative to the current frame's hand position
        '''
        self.dexycb_root = dexycb_root
        self.split = split
        self.split_sequences = get_split(dexycb_root, 's0', split)
        self.relative_norm = relative_norm
        self.rel_per_frame = rel_per_frame


        # Get all directories within directories in dexycb_root
        subjects = sorted([p for p in os.listdir(self.dexycb_root) if 'subject' in p])
        assert len(subjects) == 10
        
        self.data = []
        for subject in subjects:
            sequences = sorted(os.listdir(os.path.join(self.dexycb_root, subject)))
            assert len(sequences) == 100
            for sequence in sequences:
                if sequence not in self.split_sequences:
                    continue
                self.data.append(os.path.join(self.dexycb_root, subject, sequence))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        hand_joint_shape = (21, 3)
        pose_y_shape = (3,4)
        base_path = self.data[index]
        # print('BASE PATH:', base_path)
        frame_paths = [os.path.join(base_path, CAMERA_ID, f) for f in sorted(os.listdir(os.path.join(base_path, CAMERA_ID))) if f[:6]=='color_']
        frames = []
        
        gt_obj_quats, gt_obj_rots, gt_hand_joints, obj_class, hand_side = [], [], [], None, None
        init_hand_pose, first_hand_pose = [], None
        with open(os.path.join(base_path, 'meta.yml')) as f:
            annot = yaml.safe_load(f)
        grasp_ind = annot['ycb_grasp_ind']
        grabbing_object_id = annot['ycb_ids'][annot['ycb_grasp_ind']]
        obj_class = YCB_CLASSES[grabbing_object_id]
        obj_mesh_path = os.path.join(self.dexycb_root, 'models', obj_class, 'textured_simple.obj')
        mesh = trimesh.load(obj_mesh_path)
        # scale mesh to fit within unit cube
        rescale = max(mesh.extents)/2.
        matrix = np.eye(4)
        matrix[:3, :3] *= 1/rescale
        mesh.apply_transform(matrix)
        # randomly select N points on the mesh
        N = 100
        obj_mesh_points = mesh.sample(N)

        label_paths = [os.path.join(base_path, CAMERA_ID, f) for f in sorted(os.listdir(os.path.join(base_path, CAMERA_ID))) if f[:7]=='labels_']
        for i, label_path in enumerate(label_paths):
            label = np.load(label_path)
            pose_y = label['pose_y'][grasp_ind]
            pose_m = label['pose_m']
            
            rot = R.from_matrix(pose_y[:3, :3])
            q = np.float32(rot.as_quat())
            
            hand_exists = not np.all(pose_m == 0.0)
            obj_exists = not np.all(pose_y == 0.0)
            if not hand_exists or not obj_exists:
                continue
            
            if first_hand_pose is None:
                first_hand_pose = pose_m[0, -3:]

            assert first_hand_pose is not None

            frames.append(cv2.imread(frame_paths[i])) 

            mano_sides = annot['mano_sides']
            assert len(mano_sides) == 1
            hand_side = mano_sides[0]
            joints = label['joint_3d'].squeeze(0)
            gt_hand_joints.append(pose_m[0, -3:])
            gt_obj_quats.append(np.concatenate([q, pose_y[:3, 3]]))
            gt_obj_rots.append(pose_y)
            if self.rel_per_frame:
                init_hand_pose.append(pose_m[0, -3:])
            else:
                init_hand_pose.append(first_hand_pose)

        gt_hand_joints = np.array(gt_hand_joints)
        gt_obj_quats = np.array(gt_obj_quats)
        gt_obj_rots = np.array(gt_obj_rots)
        init_hand_pose = np.array(init_hand_pose)

        if self.relative_norm:
            gt_hand_joints -= init_hand_pose
            gt_obj_quats[:, -3:] -= init_hand_pose
            gt_obj_rots[..., 3] -= init_hand_pose

        return {
            'video_id': index,
            'frames': np.array(frames),
            'gt_hand': gt_hand_joints,
            'gt_obj': gt_obj_quats,
            'gt_obj_rot': gt_obj_rots,
            'init_hand_pose': init_hand_pose,
            'obj_class': obj_class,
            'hand_side': hand_side,
            'base_path': base_path,
            'dexycb_root': self.dexycb_root,
            'sample_points': obj_mesh_points,
        }
    
    @staticmethod
    def collate_fn(batch):
        collated_batch = {}
        min_frames = min([d['gt_obj'].shape[0] for d in batch])
        for key in batch[0].keys():
            if key == 'hand_side' or key == 'obj_class' or key == 'video_id' or key == 'base_path' or key == 'dexycb_root':
                collated_batch[key] = [d[key] for d in batch]
            elif key == 'sample_points':
                collated_batch[key] = torch.from_numpy(np.stack([d[key] for d in batch]))
            else:
                collated_batch[key] = torch.from_numpy(np.stack([d[key][-min_frames:] for d in batch]))
        return collated_batch