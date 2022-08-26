import subprocess
import argparse
import yaml
from munch import Munch
from glob import glob
import numpy as np
import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

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


def get_update_parameters(before, after, update_frac):
    changes = [np.reshape(np.abs(after[v].cpu().numpy() - before[v].cpu().numpy()), (-1,)) for v in before.keys()]
    changes = np.concatenate(changes, axis=0)
    threshold = np.percentile(changes, 100 * (1 - update_frac))
    update_params = {}
    for v in before.keys():
        abs_diff = torch.abs(after[v] - before[v])
        diff = after[v] - before[v]
        if len(abs_diff.shape) == 4:
            a, b, c, d = torch.where(abs_diff > threshold)
            a = a.cpu().detach().numpy().tolist()
            b = b.cpu().detach().numpy().tolist()
            c = c.cpu().detach().numpy().tolist()
            d = d.cpu().detach().numpy().tolist()
            coords = list(zip(a, b, c, d))
            actual_values = diff[a, b, c, d].cpu().detach().numpy().tolist()
            ret = list(zip(actual_values, coords))
            update_params[v] = ret
        elif len(diff.shape) == 1:
            coords = torch.where(abs_diff > threshold)[0].cpu().detach().numpy().tolist()
            actual_values = diff[coords].cpu().detach().numpy().tolist()
            ret = list(zip(actual_values, coords))
            update_params[v] = ret
    return update_params


def threshold_output(my_output):
    shape = my_output.shape
    cut_negative_values = (torch.ones(shape) * (-1)).to(device)
    cut_positive_values = torch.ones(shape).to(device)
    my_output = torch.maximum(my_output, cut_negative_values)
    my_output = torch.minimum(my_output, cut_positive_values)
    return my_output