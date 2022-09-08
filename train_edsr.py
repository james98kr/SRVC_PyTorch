from utils.video_reader import *
from utils.functions import *
from model.edsr import *
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio
import cv2
import os
from collections import OrderedDict


edsr_args = {'n_resblocks': 16,
            'n_feats': 64,
            'scale': [4],
            'rgb_range': 255,
            'n_colors': 3,
            'res_scale': 1}


def train():
    # CUDA, GPU setting
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')

    # Get all configurations and set up model
    cfg = get_configs()
    edsr = EDSR(edsr_args)
    edsr.to(device)
    edsr.train()

    # Declare optimizer and loss function
    optimizer = torch.optim.Adam(edsr.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    criterion = nn.MSELoss() if not cfg.L1_loss else nn.L1Loss()
    psnr_calculate = PeakSignalNoiseRatio().to(device)

    for lr_video, hr_video in zip(cfg.lr_video_list, cfg.hr_video_list):
        video_basename = os.path.basename(hr_video).split('.')[0]

        ######### In order to exclude some files from the testing process, insert their basename here #########
        if video_basename not in ['vimeo_204439769']:
            continue
        ######### In order to exclude some files from the testing process, insert their basename here #########

        print("-----------------------Start training for video %s-----------------------" % (video_basename))
        ord = OrderedDict()
        lr_cap = VideoCaptureYUV(lr_video, cfg.lr_size)
        hr_cap = VideoCaptureYUV(hr_video, cfg.hr_size)
        fps = find_video_fps(video_basename, cfg.original_path)

        for segment in range(cfg.segment_num):
            # Fetch input and output frames from capture
            try: 
                input_frames = get_segment_frames(lr_cap, cfg.segment_length * fps)
                output_frames = get_segment_frames(hr_cap, cfg.segment_length * fps)
            except:
                input_frames = []
                output_frames = []
            assert len(input_frames) == len(output_frames)

            # Skip some of the first segments
            if segment < cfg.segment_skip:
                continue
            # Save initial model parameters before update
            if segment > cfg.segment_skip:
                before = edsr.state_dict()
                before = OrderedDict({v: before[v].clone() for v in before.keys()})
            else:
                before = None

            # Perform training and find the parameters to update
            for epoch in range(cfg.epoch):
                for i, (input_frame, output_frame) in enumerate(zip(input_frames, output_frames)):
                    input_frame = input_frame.to(torch.float32).to(device) / 255
                    output_frame = output_frame.to(torch.float32).to(device) / 255
                    optimizer.zero_grad()
                    my_output_frame = edsr(input_frame).to(torch.float32)
                    loss = criterion(my_output_frame, output_frame)

                    output_frame = (output_frame * 255).to(torch.uint8)
                    my_output_frame = (my_output_frame * 255).to(torch.uint8)
                    psnr = psnr_calculate(my_output_frame, output_frame)
                    if i % 10 == 0:
                        print("Epoch: %d/%d, Video: %s, Segment: %d/%d, Frame: %d/%d, L2Loss: %f, PSNR: %f" \
                            % (epoch + 1, cfg.epoch, video_basename, segment + 1, cfg.segment_num, i, len(input_frames), loss.cpu().detach().numpy(), psnr)) 
                    loss.backward()
                    optimizer.step()

            # Save the newly trained model parameters, find parameters to update, 
            # perform one iteration of training, and update only those parameters
            if before is not None and cfg.update_frac < 1:
                after = edsr.state_dict()
                after = OrderedDict({v: after[v].clone() for v in after.keys()})
                train_mask = find_train_mask(before, after, cfg.update_frac)
                compressed = get_update_parameters(before, after, cfg.update_frac)

                # Find Delta_t by finding the change in parameters, save compressed version in OrderedDict
                delta = OrderedDict({v: (after[v] - before[v]) for v in after.keys()})
                masked_delta = OrderedDict({v: delta[v] * train_mask[v] for v in after.keys()})
                ord[segment] = compressed

                # Update parameters
                updated_params = OrderedDict({v: before[v] + masked_delta[v] for v in after.keys()})
                edsr.load_state_dict(updated_params)
                print("New parameters loaded to model. Updated fraction of parameters: %.2f" % (cfg.update_frac))
            else:
                temp = edsr.state_dict()
                ord[segment] = OrderedDict({v: temp[v].clone() for v in temp.keys()})

        torch.save(ord, "%sedsr_%s_crf%d_F%d_seg%d_skip%d_frac%.2f.pth" % (cfg.save_path, video_basename, cfg.crf, cfg.F, cfg.segment_length, cfg.segment_skip, cfg.update_frac))
        print("Saved the SRVC model for %s video file" % (video_basename))


if __name__ == "__main__":
    train()