import torch
import numpy as np
from sklearn.metrics import average_precision_score
from typing import Any, Optional, Tuple
from iopath.common.file_io import g_pathmgr

def read_label_map(label_map_file: str) -> Tuple:
        """
        Read label map and class ids.
        Args:
            label_map_file (str): Path to a .pbtxt containing class id's
                and class names
        Returns:
            (tuple): A tuple of the following,
                label_map (dict): A dictionary mapping class id to
                    the associated class names.
                class_ids (set): A set of integer unique class id's
        """
        label_map = {}
        class_ids = set()
        name = ""
        class_id = ""
        with g_pathmgr.open(label_map_file, "r") as f:
            for line in f:
                if line.startswith("  name:"):
                    name = line.split('"')[1]
                elif line.startswith("  id:") or line.startswith("  label_id:"):
                    class_id = int(line.strip().split(" ")[-1])
                    label_map[class_id] = name
                    class_ids.add(class_id)
        return label_map, class_ids

def compute_mAP(csv_file=""):
    data = np.loadtxt(csv_file, delimiter=",")
    label_map, class_ids = read_label_map("/datasets/ava_2024-01-05_0047/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt")
    person_indices = [
        index - 1 for index, name in label_map.items()
        if "person" in name.lower() or "hand shake" in name.lower()
    ]
    non_person_indices = [
        index - 1 for index, name in label_map.items()
        if "person" not in name.lower() and "hand shake" not in name.lower()
    ]
    # combine person and non person indices
    class_exists_indices = person_indices + non_person_indices
    class_exists_indices = sorted(class_exists_indices)

    num_classes = 80
    gt = data[:, :num_classes]
    pred = data[:, num_classes:]

    per_class_aps = []
    for c in range(num_classes):
        gt_c = gt[:, c]
        pred_c = pred[:, c]

        ap = average_precision_score(gt_c, pred_c)
        per_class_aps.append(ap)
    
    mAP = average_precision_score(gt[:, class_exists_indices], pred[:, class_exists_indices])
    mAP_person = average_precision_score(gt[:, person_indices], pred[:, person_indices])
    mAP_non_person = average_precision_score(gt[:, non_person_indices], pred[:, non_person_indices])


    return mAP, per_class_aps, mAP_person, mAP_non_person

if __name__ == "__main__":
    label_map, class_ids = read_label_map("/datasets/ava_2024-01-05_0047/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt")
    compute_mAP()