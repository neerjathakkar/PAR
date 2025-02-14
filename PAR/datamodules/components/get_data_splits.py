import json
import numpy as np
import os
from tqdm import tqdm
from PAR.datamodules.components.dexycb_utils import *

HOME_DIR = os.environ['HOME']


def get_split(dexycb_root, s, split):
    json_path = os.path.join(dexycb_root, 'annotation', '%s_%s.json' % (s, split))
    with open(json_path, 'r') as f:
        annot_list = json.load(f)
    sequences = []
    for idx, annot in enumerate(tqdm(annot_list)):
        example_str, camera, frame_fname = annot['color_file'].split('/')[6:]
        if camera != CAMERA_ID:
            continue
        if example_str not in sequences:
            sequences.append(example_str)
        # NOTE: for counting frames

    return sequences


def get_all_splits():
    dexycb_root = os.path.join(HOME_DIR, 'datasets/dexycb')
    s = 's0'
    train_sequences = get_split(dexycb_root, s, 'train')
    print('Train sequences:', len(train_sequences))

    val_sequences = get_split(dexycb_root, s, 'val')
    print('Val sequences:', len(val_sequences))

    test_sequences = get_split(dexycb_root, s, 'test')
    print('Test sequences:', len(test_sequences))

    return train_sequences, val_sequences, test_sequences


if __name__=='__main__':
    get_all_splits()
