import re 
from iopath.common.file_io import g_pathmgr
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type
from torch.utils.data import Dataset
from collections import defaultdict
import torch

AVA_VALID_FRAMES = list(range(902, 1799))
FPS = 30
AVA_VIDEO_START_SEC = 900

def clean_label(text):
    # Remove text within parentheses (including the parentheses)
    return re.sub(r'\s*\([^)]*\)', '', text)

def load_and_parse_labels_csv(
        frame_labels_file: str,
        # video_name_to_idx: dict,
        allowed_class_ids: Optional[Set] = None,
    ):
        """
        Parses AVA per frame labels .csv file.
        Args:
            frame_labels_file (str): Path to the file containing labels
                per key frame. Acceptible file formats are,
                Type 1:
                    <original_vido_id, frame_time_stamp, bbox_x_1, bbox_y_1, ...
                    bbox_x_2, bbox_y_2, action_lable, detection_iou>
                Type 2:
                    <original_vido_id, frame_time_stamp, bbox_x_1, bbox_y_1, ...
                    bbox_x_2, bbox_y_2, action_lable, person_label>
            video_name_to_idx (dict): Dictionary mapping video names to indices.
            allowed_class_ids (set): A set of integer unique class (bbox label)
                id's that are allowed in the dataset. If not set, all class id's
                are allowed in the bbox labels.
        Returns:
            (dict): A dictionary of dictionary containing labels per each keyframe
                in each video. Here, the label for each keyframe is again a dict
                of the form,
                {
                    'labels': a list of bounding boxes
                    'boxes':a list of action lables for the bounding box
                    'extra_info': ist of extra information cotaining either
                        detections iou's or person id's depending on the
                        csv format.
                }
        """
        labels_dict = {}
        video_name_to_idx = {}
        video_idx_to_name = {}
        with g_pathmgr.open(frame_labels_file, "r") as f:
            for line in f:
                row = line.strip().split(",")
                video_name = row[0]
                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name[idx] = video_name
                video_idx = video_name_to_idx[video_name] 

                frame_sec = float(row[1])
                if (
                    frame_sec > AVA_VALID_FRAMES[-1]
                    or frame_sec < AVA_VALID_FRAMES[0]
                ):
                    continue

                # Since frame labels in video start from 0 not at 900 secs
                frame_sec = frame_sec - AVA_VIDEO_START_SEC

                # Box with format [x1, y1, x2, y2] with a range of [0, 1] as float.
                bbox = list(map(float, row[2:6]))

                # Label
                label = -1 if row[6] == "" else int(row[6])
                # Continue if the current label is not in allowed labels.
                if (allowed_class_ids is not None) and (label not in allowed_class_ids):
                    continue

                # Both id's and iou's are treated as float
                extra_info = float(row[7])

                if video_idx not in labels_dict:
                    labels_dict[video_idx] = {}

                if frame_sec not in labels_dict[video_idx]:
                    labels_dict[video_idx][frame_sec] = defaultdict(list)

                labels_dict[video_idx][frame_sec]["boxes"].append(bbox)
                labels_dict[video_idx][frame_sec]["labels"].append(label)
                labels_dict[video_idx][frame_sec]["extra_info"].append(extra_info)
        return labels_dict, video_name_to_idx, video_idx_to_name

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
    


def extract_tracks(labels_dict, video_name_to_idx, N=4, stride=2, max_track_len=8):
    results = []
    for video_idx, timesteps in labels_dict.items():
        track_data = {}

        # collect tracks by timestep
        for time, data in timesteps.items():
            for idx, track_id in enumerate(data["extra_info"]):
                if track_id not in track_data:
                    track_data[track_id] = {}
                if time not in track_data[track_id]:
                    track_data[track_id][time] = {
                        "bbox": data["boxes"][idx],
                        "labels": [data["labels"][idx]]
                    }
                else:
                    track_data[track_id][time]["labels"].append(data["labels"][idx])
                    track_data[track_id][time]["bbox"] = data["boxes"][idx]

        for track_id, track_entries in track_data.items():
            if len(track_entries) >= N:
                agent_timesteps = list(track_entries.keys())

                # find other tracks at these timesteps  and add to track_data
                for t in agent_timesteps:
                    for idx, t_id in enumerate(timesteps[t]["extra_info"]):
                        if t_id != track_id:
                            bbox = track_data[t_id][t]["bbox"]
                            labels = track_data[t_id][t]["labels"]

                            if "other_agents" not in track_data[track_id][t]:
                                track_data[track_id][t]["other_agents"] = {t_id: {"bbox": bbox, "labels": labels}}
                            else:
                                track_data[track_id][t]["other_agents"][t_id] = {"bbox": bbox, "labels": labels}
                if len(track_data[track_id].keys()) >= max_track_len + stride:
                    all_timesteps = list(track_data[track_id].keys())
                    subsequences = []
                    for i in range(0, len(all_timesteps) - max_track_len + 1, stride):
                        subsequences.append(all_timesteps[i:i + max_track_len])
                    
                    for subseq in subsequences:
                        agent_presence_freq = {}
                        # copy track data at timesteps
                        track_data_subseq = {}
                        track_data_subseq[track_id] = {t: track_data[track_id][t] for t in subseq}

                        for timestep, data in track_entries.items():
                            if timestep in subseq and "other_agents" in track_data[track_id][timestep]:
                                for other_id, other_data in track_data[track_id][timestep]["other_agents"].items():
                                    if other_id not in agent_presence_freq:
                                        agent_presence_freq[other_id] = 1
                                    else:
                                        agent_presence_freq[other_id] += 1
                        sorted_agents = sorted(agent_presence_freq.items(), key=lambda x: x[1], reverse=True)
                        other_ids = [agent[0] for agent in sorted_agents]

                        track_data_subseq[track_id]["other_agents_freq_sorted"] = other_ids
                        track_data_subseq[track_id]["ego_id"] = track_id
                        track_data_subseq[track_id]["video_id"] = video_idx

                        results.append(track_data_subseq[track_id])
                else:
                    agent_presence_freq = {}
                    for timestep, data in track_entries.items():
                        if "other_agents" in data:
                            for other_id, other_data in data["other_agents"].items():
                                # print(timestep, other_id, other_data)
                                if other_id not in agent_presence_freq:
                                    agent_presence_freq[other_id] = 1
                                else:
                                    agent_presence_freq[other_id] += 1
                    sorted_agents = sorted(agent_presence_freq.items(), key=lambda x: x[1], reverse=True)
                    other_ids = [agent[0] for agent in sorted_agents]

                    track_data[track_id]["other_agents_freq_sorted"] = other_ids
                    track_data[track_id]["ego_id"] = track_id
                    track_data[track_id]["video_id"] = video_idx

                    results.append(track_data[track_id])

    # results has all tracks with at least N timesteps
    # timestamps are keys, each key has bbox and label
    # also ego_ID and video ID as keys
    print("Results: ", len(results))
    return results


class AVAActionsDataset(Dataset):
    def __init__(self, frame_labels_file, num_agents=2, 
                 min_track_length=4, max_track_length=12, stride=2, 
                 add_input_noise=False, input_noise_std=0.2,
                 get_image=False):
        self.labels_dict, self.video_name_to_idx, self.video_idx_to_name = load_and_parse_labels_csv(frame_labels_file)
        self.tracks = extract_tracks(self.labels_dict, self.video_name_to_idx, N=min_track_length, stride=stride, max_track_len=max_track_length)
        self.num_classes = 80
        self.num_agents = num_agents

        self.pad_seq_len = max_track_length
        self.add_input_noise = add_input_noise
        self.input_noise_std = input_noise_std

        self.get_image = get_image

    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        # print(self.tracks[idx])
        track = self.tracks[idx]
        ego_id = track["ego_id"]
        video_id = track["video_id"]
        freq_agents = track["other_agents_freq_sorted"]
        if len(freq_agents) < self.num_agents - 1:
            agent_pad_len = self.num_agents - 1 - len(freq_agents)
        else:
            freq_agents = freq_agents[:self.num_agents - 1]
            agent_pad_len = 0
        timesteps = list(track.keys())
        timesteps.remove("ego_id")
        timesteps.remove("video_id")
        timesteps.remove("other_agents_freq_sorted")
        timesteps.sort() 


        # will be BS, seq_len (padded), num_agents, 4
        agent_bboxes = torch.zeros(self.pad_seq_len, self.num_agents, 4)
        # will be BS, seq_len (padded), num_agent, num_classes
        agent_labels = torch.zeros(self.pad_seq_len, self.num_agents, self.num_classes)

        # iterate over all timesteps
        for i, t in enumerate(timesteps):
            if i >= self.pad_seq_len:
                break

            # zero index labels and convert to an 80D tensor
            labels_0_idx = []
            for label in track[t]["labels"]:
                labels_0_idx.append(label - 1)
            labels_tensor = torch.zeros(self.num_classes)
            labels_tensor[labels_0_idx] = 1.0
            if self.add_input_noise:
                labels_tensor = labels_tensor + torch.randn(self.num_classes) * self.input_noise_std
                

            # ego agent should be last
            agent_bboxes[i, -1, :] = torch.Tensor(track[t]["bbox"])
            agent_labels[i, -1, :] = labels_tensor

            # iterate over as many other agents as we have
            for k in range(self.num_agents - 1 - agent_pad_len):
                other_id = freq_agents[k]
                if "other_agents" in track[t] and other_id in track[t]["other_agents"]:
                    other_data = track[t]["other_agents"][other_id]
                    agent_bboxes[i, k, :] = torch.Tensor(other_data["bbox"])
                    labels_0_idx = []
                    for label in other_data["labels"]:
                        labels_0_idx.append(label - 1)
                    labels_tensor = torch.zeros(self.num_classes)
                    labels_tensor[labels_0_idx] = 1.0
                    if self.add_input_noise:
                        labels_tensor = labels_tensor + torch.randn(self.num_classes) * self.input_noise_std
                    agent_labels[i, k, :] = labels_tensor
                # agent not present at timestep, so pad
                else:
                    agent_bboxes[i, k, :] = torch.zeros(4)
                    agent_labels[i, k, :] = torch.zeros(self.num_classes)
            
            # if not enough other agents, pad with zeros
            for j in range(self.num_agents - 1 - agent_pad_len, self.num_agents - 1):
                agent_bboxes[i, j, :] = torch.zeros(4)
                agent_labels[i, j, :] = torch.zeros(self.num_classes)

        if len(timesteps) < self.pad_seq_len:
            true_seq_len = len(timesteps)
            for p in range(true_seq_len, self.pad_seq_len):
                for n in range(self.num_agents):
                    agent_bboxes[p, n, :] = torch.zeros(4)
                    agent_labels[p, n, :] = torch.zeros(self.num_classes)

        return {
            "agent_bboxes": agent_bboxes,
            "agent_labels": agent_labels,
            "video_id": video_id,
            "ego_id": ego_id,
            # "all_ids": freq_agents + agent_pad_len*[-1] + [ego_id],
            "video_name": self.video_idx_to_name[video_id], 
            "start_timestep": timesteps[0]
        }
