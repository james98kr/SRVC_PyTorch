import subprocess
import argparse
import yaml
from munch import Munch
from glob import glob
import numpy as np
import torch

def parse_size(string):
    temp = string.split(',')
    return (int(temp[0]), int(temp[1]))


def find_video_fps(video_basename, original_path):
    fpscmd = "ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate " + original_path + video_basename + ".mp4"
    result = subprocess.run(fpscmd.split(' '), stdout=subprocess.PIPE).stdout.decode('utf-8')[:-1]
    if len(result.split('/')) == 2:
        fps = float(result.split('/')[0]) / float(result.split('/')[1])
        fps = int(round(fps))
    else:
        fps = int(round(float(result)))
    return fps


def get_configs():
    parser = argparse.ArgumentParser('srvc')
    parser.add_argument('config', type=str, help='path to config file')
    args = parser.parse_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    cfg.lr_video_list = sorted(glob(cfg.lr_path + '*crf' + str(cfg.crf) + '*'))
    cfg.hr_video_list = sorted(glob(cfg.hr_path + '*'))[:len(cfg.lr_video_list)]
    cfg.saved_file_list = sorted(glob(cfg.save_path + '*' + str(cfg.crf) + '*' + str(cfg.F) + '*' + str(cfg.segment_length) + '*'))
    cfg.hr_size = parse_size(cfg.hr_size)
    cfg.lr_size = parse_size(cfg.lr_size)
    cfg.patch_size = parse_size(cfg.patch_size)
    cfg.video_length = cfg.end_time - cfg.start_time
    cfg.segment_num = cfg.video_length // cfg.segment_length
    return cfg


def find_train_mask(before, after, update_frac):
    changes = [np.reshape(np.abs(after[v].cpu().numpy() - before[v].cpu().numpy()), (-1,)) for v in before.keys()]
    changes = np.concatenate(changes, axis=0)
    threshold = np.percentile(changes, 100 * (1 - update_frac))
    train_mask = {v: torch.abs(after[v] - before[v]) > threshold for v in before.keys()}
    return train_mask