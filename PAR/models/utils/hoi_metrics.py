import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
import point_cloud_utils as pcu

from typing import Optional, Dict, List, Tuple


###############################################################################
# Object metrics
###############################################################################
def sample_points(v, f, num_samples=10000):
    """
    Randomly sample points on the surface of triangle mesh.
    
    Args:
        v (np.array or Tensor): Triangle mesh vertices (V, 3).
        f (np.array or Tensor): Triangle mesh faces (F, 3).
    """
    if torch.is_tensor(v):
        v = v.numpy()
        f = f.numpy()
    fid, bc = pcu.sample_mesh_random(v, f, num_samples)
    # Interpolate the vertex positions and normals using the returned barycentric coordinates
    # to get sample positions and normals
    obj_verts = pcu.interpolate_barycentric_coords(f, fid, bc, v)
    return obj_verts


def fscore(
    pred_pts: torch.Tensor,
    gt_pts: torch.Tensor,
    num_samples: int = 10000,
    th_list: List = [0.005, 0.01]
) -> List:
    """
    Batch computes F1-score between two point sets (of the same size).

    Args:
        pred_pts (Tensor): First point cloud of shape (B, N, 3).
        gt_pts (Tensor): Second point cloud of shape (B, N, 3).
        num_samples (int): Number of particles per point cloud.
        th_list (List): List of Chamfer Distance thresholds in meters.

    """
    (d1, d2), _ = chamfer_distance(pred_pts, gt_pts, batch_reduction=None, point_reduction=None)
    d1 = torch.sqrt(d1)
    d2 = torch.sqrt(d2)

    res_list = []
    for th in th_list:
        if d1.size(1) and d2.size(1):
            recall = torch.sum(d2 < th, dim=-1).to(gt_pts) / num_samples  # recall knn(gt, pred) gt->pred
            precision = torch.sum(d1 < th, dim=-1).to(gt_pts) / num_samples  # precision knn(pred, gt) pred-->gt
            eps = 1e-8
            fscore = 2 * recall * precision / (recall + precision + eps)
            res_list.append(fscore.tolist())
        else:
            raise ValueError("d1 and d2 should be in equal length but got %d %d" % (d1.size(1), d2.size(1)))
    # Add CD
    d = ((d1 ** 2).mean(1) + (d2 ** 2).mean(1)).tolist()
    return res_list + [d, ]


def cdscore(
    pred_pts: torch.Tensor,
    gt_pts: torch.Tensor,
    num_samples: int = 10000
) -> torch.Tensor:
    """
    Batch computes Chamfer Distance between two point sets (of the same size).
    
    Args:
        pred_pts (Tensor): First point cloud of shape (B, N, 3).
        gt_pts (Tensor): Second point cloud of shape (B, N, 3).
        num_samples (int): Number of particles per point cloud.
    """
    pred_pts_mm = pred_pts
    gt_pts_mm = gt_pts
    assert len(pred_pts_mm[0]) == len(gt_pts_mm[0]) == num_samples
    (d1, d2), _ = chamfer_distance(pred_pts_mm, gt_pts_mm, batch_reduction=None, point_reduction=None)
    d = ((d1).mean(1) + (d2).mean(1)).tolist()
    return d


###############################################################################
# Hand metrics
# Code adapted from: https://github.com/geopavlakos/hamer/hamer/utils/pose_utils.py
###############################################################################
def get_pck(kp_id, threshold, euclidean_dist):
    """ Returns pck for one keypoint for the given threshold. """
    data = np.array(euclidean_dist[kp_id])
    pck = np.mean((data <= threshold).astype('float'))
    return pck


def auc(gt_joints, pred_joints):
    """
    For each thresholds, computes thresh_ious and averages accross batch dim
    """
    diff = gt_joints - pred_joints
    euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))
    assert len(euclidean_dist) == 21

    val_min = 0.0
    val_max = 0.05 #m
    steps = 100
    thresholds = np.linspace(val_min, val_max, steps)
    thresholds = np.array(thresholds)

    norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

    # init mean measures
    auc_all = list()
    #pck_curve_all = list()

    # Create one plot for each part
    num_kp = len(gt_joints)
    for part_id in range(num_kp):
        # pck/auc
        pck_curve = list()
        for t in thresholds:
            pck = get_pck(part_id, t, euclidean_dist)
            pck_curve.append(pck)

        pck_curve = np.array(pck_curve)
        #pck_curve_all.append(pck_curve)
        auc = np.trapz(pck_curve, thresholds)
        auc /= norm_factor
        auc_all.append(auc)

    return np.mean(auc_all)


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)

def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt( ((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
    return re

def eval_pose(pred_joints, gt_joints) -> Tuple[np.array, np.array]:
    """
    Compute joint errors in cm before and after Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    Returns:
        Tuple[np.array, np.array]: Joint errors in cm before and after alignment.
    """
    # Absolute error (MPJPE)
    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    # Reconstruction_error
    r_error = reconstruction_error(pred_joints, gt_joints).cpu().numpy()
    return 100 * mpjpe, 100 * r_error

class Evaluator:

    def __init__(self,
                 dataset_length: int,
                 keypoint_list: List,
                 pelvis_ind: int,
                 metrics: List = ['mode_mpjpe', 'mode_re', 'min_mpjpe', 'min_re'],
                 pck_thresholds: Optional[List] = None):
        """
        Class used for evaluating trained models on different 3D pose datasets.
        Args:
            dataset_length (int): Total dataset length.
            keypoint_list [List]: List of keypoints used for evaluation.
            pelvis_ind (int): Index of pelvis keypoint; used for aligning the predictions and ground truth.
            metrics [List]: List of evaluation metrics to record.
        """
        self.dataset_length = dataset_length
        self.keypoint_list = keypoint_list
        self.pelvis_ind = pelvis_ind
        self.metrics = metrics
        for metric in self.metrics:
            setattr(self, metric, np.zeros((dataset_length,)))
        self.counter = 0
        if pck_thresholds is None:
            self.pck_evaluator = None
        else:
            self.pck_evaluator = EvaluatorPCK(pck_thresholds)

    def log(self):
        """
        Print current evaluation metrics
        """
        if self.counter == 0:
            print('Evaluation has not started')
            return
        print(f'{self.counter} / {self.dataset_length} samples')
        if self.pck_evaluator is not None:
            self.pck_evaluator.log()
        for metric in self.metrics:
            if metric in ['mode_mpjpe', 'mode_re', 'min_mpjpe', 'min_re']:
                unit = 'mm'
            else:
                unit = ''
            print(f'{metric}: {getattr(self, metric)[:self.counter].mean()} {unit}')
        print('***')

    def get_metrics_dict(self) -> Dict:
        """
        Returns:
            Dict: Dictionary of evaluation metrics.
        """
        d1 = {metric: getattr(self, metric)[:self.counter].mean() for metric in self.metrics}
        if self.pck_evaluator is not None:
            d2 = self.pck_evaluator.get_metrics_dict()
            d1.update(d2)
        return d1

    def __call__(self, output: Dict, batch: Dict, opt_output: Optional[Dict] = None):
        """
        Evaluate current batch.
        Args:
            output (Dict): Regression output.
            batch (Dict): Dictionary containing images and their corresponding annotations.
            opt_output (Dict): Optimization output.
        """
        if self.pck_evaluator is not None:
            self.pck_evaluator(output, batch, opt_output)

        pred_keypoints_3d = output['pred_keypoints_3d'].detach()
        pred_keypoints_3d = pred_keypoints_3d[:,None,:,:]
        batch_size = pred_keypoints_3d.shape[0]
        num_samples = pred_keypoints_3d.shape[1]
        gt_keypoints_3d = batch['keypoints_3d'][:, :, :-1].unsqueeze(1).repeat(1, num_samples, 1, 1)

        # Align predictions and ground truth such that the pelvis location is at the origin
        pred_keypoints_3d -= pred_keypoints_3d[:, :, [self.pelvis_ind]]
        gt_keypoints_3d -= gt_keypoints_3d[:, :, [self.pelvis_ind]]

        # Compute joint errors
        mpjpe, re = eval_pose(pred_keypoints_3d.reshape(batch_size * num_samples, -1, 3)[:, self.keypoint_list], gt_keypoints_3d.reshape(batch_size * num_samples, -1 ,3)[:, self.keypoint_list])
        mpjpe = mpjpe.reshape(batch_size, num_samples)
        re = re.reshape(batch_size, num_samples)

        # Compute 2d keypoint errors
        pred_keypoints_2d = output['pred_keypoints_2d'].detach()
        pred_keypoints_2d = pred_keypoints_2d[:,None,:,:]
        gt_keypoints_2d = batch['keypoints_2d'][:,None,:,:].repeat(1, num_samples, 1, 1)
        conf = gt_keypoints_2d[:, :, :, -1].clone()
        kp_err = torch.nn.functional.mse_loss(
                        pred_keypoints_2d,
                        gt_keypoints_2d[:, :, :, :-1],
                        reduction='none'
                    ).sum(dim=3)
        kp_l2_loss = (conf * kp_err).mean(dim=2)
        kp_l2_loss = kp_l2_loss.detach().cpu().numpy()

        # Compute joint errors after optimization, if available.
        if opt_output is not None:
            opt_keypoints_3d = opt_output['model_joints']
            opt_keypoints_3d -= opt_keypoints_3d[:, [self.pelvis_ind]]
            opt_mpjpe, opt_re = eval_pose(opt_keypoints_3d[:, self.keypoint_list], gt_keypoints_3d[:, 0, self.keypoint_list])

        # The 0-th sample always corresponds to the mode
        if hasattr(self, 'mode_mpjpe'):
            mode_mpjpe = mpjpe[:, 0]
            self.mode_mpjpe[self.counter:self.counter+batch_size] = mode_mpjpe
        if hasattr(self, 'mode_re'):
            mode_re = re[:, 0]
            self.mode_re[self.counter:self.counter+batch_size] = mode_re
        if hasattr(self, 'mode_kpl2'):
            mode_kpl2 = kp_l2_loss[:, 0]
            self.mode_kpl2[self.counter:self.counter+batch_size] = mode_kpl2
        if hasattr(self, 'min_mpjpe'):
            min_mpjpe = mpjpe.min(axis=-1)
            self.min_mpjpe[self.counter:self.counter+batch_size] = min_mpjpe
        if hasattr(self, 'min_re'):
            min_re = re.min(axis=-1)
            self.min_re[self.counter:self.counter+batch_size] = min_re
        if hasattr(self, 'min_kpl2'):
            min_kpl2 = kp_l2_loss.min(axis=-1)
            self.min_kpl2[self.counter:self.counter+batch_size] = min_kpl2
        if hasattr(self, 'opt_mpjpe'):
            self.opt_mpjpe[self.counter:self.counter+batch_size] = opt_mpjpe
        if hasattr(self, 'opt_re'):
            self.opt_re[self.counter:self.counter+batch_size] = opt_re

        self.counter += batch_size

        if hasattr(self, 'mode_mpjpe') and hasattr(self, 'mode_re'):
            return {
                'mode_mpjpe': mode_mpjpe,
                'mode_re': mode_re,
            }
        else:
            return {}


class EvaluatorPCK:

    def __init__(self, thresholds: List = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],):
        """
        Class used for evaluating trained models on different 3D pose datasets.
        Args:
            thresholds [List]: List of PCK thresholds to evaluate.
            metrics [List]: List of evaluation metrics to record.
        """
        self.thresholds = thresholds
        self.pred_kp_2d = []
        self.gt_kp_2d = []
        self.gt_conf_2d = []
        self.counter = 0

    def log(self):
        """
        Print current evaluation metrics
        """
        if self.counter == 0:
            print('Evaluation has not started')
            return
        print(f'{self.counter} samples')
        metrics_dict = self.get_metrics_dict()
        for metric in metrics_dict:
            print(f'{metric}: {metrics_dict[metric]}')
        print('***')

    def get_metrics_dict(self) -> Dict:
        """
        Returns:
            Dict: Dictionary of evaluation metrics.
        """
        pcks = self.compute_pcks()
        metrics = {}
        for thr, (acc,avg_acc,cnt) in zip(self.thresholds, pcks):
            metrics.update({f'kp{i}_pck_{thr}': float(a) for i, a in enumerate(acc) if a>=0})
            metrics.update({f'kpAvg_pck_{thr}': float(avg_acc)})
        return metrics

    def compute_pcks(self):
        pred_kp_2d = np.concatenate(self.pred_kp_2d, axis=0)
        gt_kp_2d = np.concatenate(self.gt_kp_2d, axis=0)
        gt_conf_2d = np.concatenate(self.gt_conf_2d, axis=0)
        assert pred_kp_2d.shape == gt_kp_2d.shape
        assert pred_kp_2d[..., 0].shape == gt_conf_2d.shape
        assert pred_kp_2d.shape[1] == 1 # num_samples

        from mmpose.core.evaluation import keypoint_pck_accuracy
        pcks = [
            keypoint_pck_accuracy(
                pred_kp_2d[:, 0, :, :],
                gt_kp_2d[:, 0, :, :],
                gt_conf_2d[:, 0, :]>0.5,
                thr=thr,
                normalize = np.ones((len(pred_kp_2d),2))   # Already in [-0.5,0.5] range. No need to normalize
            )
            for thr in self.thresholds
        ]
        return pcks

    def __call__(self, output: Dict, batch: Dict, opt_output: Optional[Dict] = None):
        """
        Evaluate current batch.
        Args:
            output (Dict): Regression output.
            batch (Dict): Dictionary containing images and their corresponding annotations.
            opt_output (Dict): Optimization output.
        """
        pred_keypoints_2d = output['pred_keypoints_2d'].detach()
        num_samples = 1
        batch_size = pred_keypoints_2d.shape[0]

        pred_keypoints_2d = pred_keypoints_2d[:,None,:,:]
        gt_keypoints_2d = batch['keypoints_2d'][:,None,:,:].repeat(1, num_samples, 1, 1)

        self.pred_kp_2d.append(pred_keypoints_2d[:, :, :, :2].detach().cpu().numpy())
        self.gt_conf_2d.append(gt_keypoints_2d[:, :, :, -1].detach().cpu().numpy())
        self.gt_kp_2d.append(gt_keypoints_2d[:, :, :, :2].detach().cpu().numpy())

        self.counter += batch_size


if __name__ == "__main__":
    pass
