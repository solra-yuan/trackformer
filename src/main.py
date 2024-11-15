# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
import time
from os import path as osp
import random

import cv2
from PIL import Image, ImageDraw

import motmetrics as mm
import numpy as np
import sacred
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from trackformer.datasets.coco import make_coco_transforms
from trackformer.datasets.transforms import Compose
from trackformer.datasets.tracking import TrackDatasetFactory
from trackformer.models import build_model
from trackformer.models.tracker import Tracker
from trackformer.util.misc import nested_dict_to_namespace
from trackformer.util.track_utils import (evaluate_mot_accums, get_mot_accum,
                                          interpolate_tracks, plot_sequence)

mm.lap.default_solver = 'lap'

ex = sacred.Experiment('track')
ex.add_config('cfgs/track.yaml')
ex.add_named_config('reid', 'cfgs/track_reid.yaml')

resize_size = (256, 256)

random.seed(214)

def imgToBatch(img, transforms):
    width_orig, height_orig = img.size

    img, _ = transforms(img)
    width, height = img.size(2), img.size(1)

    sample = {}
    sample['img'] = img.unsqueeze(0)
    sample['dets'] = torch.tensor([]).unsqueeze(0)
    sample['orig_size'] = torch.as_tensor(
        [int(height_orig), int(width_orig)]).unsqueeze(0)
    sample['size'] = torch.as_tensor([int(height), int(width)]).unsqueeze(0)

    return sample


@ex.automain
def main(seed, obj_detect_checkpoint_file, tracker_cfg,
         write_images, output_dir, interpolate,
         verbose, generate_attention_maps,
         _config, _log, _run, obj_detector_model=None):

    if write_images:
        assert output_dir is not None

    # obj_detector_model is only provided when run as evaluation during
    # training. in that case we omit verbose outputs.
    if obj_detector_model is None:
        sacred.commands.print_config(_run)

    # set all seeds
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    if output_dir is not None:
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        yaml.dump(
            _config,
            open(osp.join(output_dir, 'track.yaml'), 'w'),
            default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    if obj_detector_model is None:
        obj_detect_config_path = os.path.join(
            os.path.dirname(obj_detect_checkpoint_file),
            'config.yaml')
        obj_detect_args = nested_dict_to_namespace(
            yaml.unsafe_load(open(obj_detect_config_path)))
        img_transform = obj_detect_args.img_transform
        obj_detector, _, obj_detector_post = build_model(obj_detect_args)

        obj_detect_checkpoint = torch.load(
            obj_detect_checkpoint_file, map_location=lambda storage, loc: storage)

        obj_detect_state_dict = obj_detect_checkpoint['model']

        obj_detect_state_dict = {
            k.replace('detr.', ''): v
            for k, v in obj_detect_state_dict.items()
            if 'track_encoding' not in k}

        obj_detector.load_state_dict(obj_detect_state_dict)
        if 'epoch' in obj_detect_checkpoint:
            _log.info(
                f"INIT object detector [EPOCH: {obj_detect_checkpoint['epoch']}]")

        obj_detector.cuda()
    else:
        obj_detector = obj_detector_model['model']
        obj_detector_post = obj_detector_model['post']
        img_transform = obj_detector_model['img_transform']

    if hasattr(obj_detector, 'tracking'):
        obj_detector.tracking()

    track_logger = None

    if verbose:
        track_logger = _log.info

    tracker = Tracker(
        obj_detector,
        obj_detector_post,
        tracker_cfg,
        generate_attention_maps,
        track_logger,
        verbose
    )

    cap = cv2.VideoCapture('./src/test2.mp4')
    frame_skip=5
    frame_count=0
    transforms = Compose(make_coco_transforms(
        'val',
        img_transform,
        overflow_boxes=True
    ))

    color_map = {}
    prev_values = {}
    tracker.reset()

    while True:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if frame_count % frame_skip == 0:
            origin_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(origin_img, resize_size)
            img = Image.fromarray(img)

            batch = imgToBatch(img, transforms)
            with torch.no_grad():
                tracker.step(batch)
            results = tracker.get_results()       
            draw = ImageDraw.Draw(img)

            last_values = {key: sub_dict[max(sub_dict.keys())] for key, sub_dict in results.items()}
            
            filtered_results = {}
            for key2, value2 in last_values.items():
                found_same_score = False
                for key1, value1 in prev_values.items():
                    if value2['score'] == value1['score']:
                        found_same_score = True
                        break
                if not found_same_score:
                    filtered_results[key2] = value2

            for key in filtered_results:
                result = results[key]
                last_key = list(result.keys())[-1]
                last_item = result[last_key]
                if 'score' in last_item:
                    id = last_item['obj_ind']
                    if id not in color_map:                
                        random_color = (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255)
                        )
                        color_map[id] = random_color
                    else:
                        random_color = color_map[id]
                    draw.rectangle(last_item['bbox'], outline=random_color, width=2)
            
            prev_values = last_values
            torch.cuda.empty_cache()

            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imshow('test', frame)
            cv2.waitKey(1)
        frame_count += 1