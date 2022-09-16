from utils.video_reader import *
from utils.functions import *
from utils.dataset import *
from model.sr_model import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio
import os, time
from collections import OrderedDict

def train():
    # CUDA, GPU setting
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')

    # Get all configurations and set up model
    cfg = get_configs()
    sr_model = nn.DataParallel(SR_Model(cfg.F, cfg.scale, cfg.patch_size))
    sr_model.to(device)
    sr_model.train()

    # Declare optimizer and loss function
    optimizer = torch.optim.Adam(sr_model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    criterion = nn.MSELoss() if not cfg.L1_loss else nn.L1Loss()
    psnr_calculate = PeakSignalNoiseRatio(data_range=1.0).to(device)

    for lr_video, hr_video in zip(cfg.lr_video_list, cfg.hr_video_list):
        video_basename = os.path.basename(hr_video).split('.')[0]
        lr_video_basename = os.path.basename(lr_video).split('.')[0].split('_crf')[0]
        assert video_basename == lr_video_basename

        ######### In order to exclude some files from the testing process, insert their basename here #########
        if video_basename in []:
            continue
        ######### In order to exclude some files from the testing process, insert their basename here #########

        print("-----------------------Start training for video %s-----------------------" % (video_basename))
        ord = OrderedDict()
        lr_cap = cv2.VideoCapture(lr_video)
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
                before = sr_model.state_dict()
                before = OrderedDict({v: before[v].clone() for v in before.keys()})
            else:
                before = None

            # Perform one iteration of training for finding the training mask
            print("First iteration for training mask:")
            srvc_dataset = SRVC_DataSet(input_frames, output_frames)
            srvc_dataloader = DataLoader(srvc_dataset, batch_size=cfg.batch_size, shuffle=False)
            for i, (input_frame, output_frame) in enumerate(srvc_dataloader):
                if cfg.crop:
                    input_frame, output_frame = crop_frame(input_frame, output_frame)
                input_frame = input_frame.to(torch.float32).to(device) / 255
                output_frame = output_frame.to(torch.float32).to(device) / 255
                optimizer.zero_grad()
                my_output_frame = sr_model(input_frame).to(torch.float32)
                loss = criterion(my_output_frame, output_frame)
                psnr = psnr_calculate(my_output_frame, output_frame)
                print("First iteration - Segment: %d/%d, Frame: %d/%d, L2Loss: %f, PSNR: %f" \
                    % (segment + 1, cfg.segment_num, i + 1, len(srvc_dataloader), loss.cpu().detach().numpy(), psnr)) 
                loss.backward()
                optimizer.step()

            # Find training mask if segment is greater than segment_skip or cfg.update_frac < 1
            # Otherwise simply continue training, and save the model parameters to ord
            print("Actual training for Adam updates:")
            if before is None or cfg.update_frac == 1:
                for epoch in range(cfg.epoch - 1):
                    srvc_dataset = SRVC_DataSet(input_frames, output_frames)
                    srvc_dataloader = DataLoader(srvc_dataset, batch_size=cfg.batch_size, shuffle=False)
                    for i, (input_frame, output_frame) in enumerate(srvc_dataloader):
                        if cfg.crop:
                            input_frame, output_frame = crop_frame(input_frame, output_frame)
                        input_frame = input_frame.to(torch.float32).to(device) / 255
                        output_frame = output_frame.to(torch.float32).to(device) / 255
                        optimizer.zero_grad()
                        my_output_frame = sr_model(input_frame).to(torch.float32)
                        loss = criterion(my_output_frame, output_frame)
                        psnr = psnr_calculate(my_output_frame, output_frame)
                        print("Epoch: %d/%d, Video: %s, Segment: %d/%d, Frame: %d/%d, L2Loss: %f, PSNR: %f" \
                            % (epoch + 2, cfg.epoch, video_basename, segment + 1, cfg.segment_num, i + 1, len(srvc_dataloader), loss.cpu().detach().numpy(), psnr)) 
                        loss.backward()
                        optimizer.step()
                temp = sr_model.state_dict()
                ord[segment] = OrderedDict({v: temp[v].clone() for v in temp.keys()})
            else:
                after = sr_model.state_dict()
                after = OrderedDict({v: after[v].clone() for v in after.keys()})
                train_mask = find_train_mask(before, after, cfg.update_frac)
                delta = OrderedDict({v: (after[v] - before[v]) for v in after.keys()})
                masked_delta = OrderedDict({v: delta[v] * train_mask[v] for v in delta.keys()})
                updated_params = OrderedDict({v: before[v] + masked_delta[v] for v in before.keys()})
                sr_model.load_state_dict(updated_params)

                for epoch in range(cfg.epoch - 1):
                    srvc_dataset = SRVC_DataSet(input_frames, output_frames)
                    srvc_dataloader = DataLoader(srvc_dataset, batch_size=cfg.batch_size, shuffle=False)

                    # Parameters before current training iteration
                    prev = sr_model.state_dict()
                    prev = OrderedDict({v: prev[v].clone() for v in prev.keys()})

                    for i, (input_frame, output_frame) in enumerate(srvc_dataloader):
                        if cfg.crop:
                            input_frame, output_frame = crop_frame(input_frame, output_frame)
                        input_frame = input_frame.to(torch.float32).to(device) / 255
                        output_frame = output_frame.to(torch.float32).to(device) / 255
                        optimizer.zero_grad()
                        my_output_frame = sr_model(input_frame).to(torch.float32)
                        loss = criterion(my_output_frame, output_frame)
                        psnr = psnr_calculate(my_output_frame, output_frame)
                        print("Epoch: %d/%d, Video: %s, Segment: %d/%d, Frame: %d/%d, L2Loss: %f, PSNR: %f" \
                            % (epoch + 2, cfg.epoch, video_basename, segment + 1, cfg.segment_num, i + 1, len(srvc_dataloader), loss.cpu().detach().numpy(), psnr)) 
                        loss.backward()
                        optimizer.step()

                    # Parameters after current training iteration
                    next = sr_model.state_dict()
                    next = OrderedDict({v: next[v].clone() for v in next.keys()})

                    # Calculate delta and masked_delta, and update the model parameters
                    delta = OrderedDict({v: (next[v] - prev[v]) for v in next.keys()})
                    masked_delta = OrderedDict({v: delta[v] * train_mask[v] for v in delta.keys()})
                    updated_params = OrderedDict({v: prev[v] + masked_delta[v] for v in prev.keys()})
                    sr_model.load_state_dict(updated_params)

                final = sr_model.state_dict()
                final = OrderedDict({v: final[v].clone() for v in final.keys()})
                delta = OrderedDict({v: (final[v] - before[v]) for v in before.keys()})
                masked_delta = OrderedDict({v: delta[v] * train_mask[v] for v in delta.keys()})
                updated_params = OrderedDict({v: before[v] + masked_delta[v] for v in before.keys()})
                sr_model.load_state_dict(updated_params)
                compressed = get_update_parameters(before, final, cfg.update_frac)
                ord[segment] = compressed
                cnt = 0
                for v in compressed.keys():
                    cnt += len(compressed[v][0])
                print("Segment: %d/%d, updated fraction of parameters: %.2f, updated number of parameters: %d" % (segment + 1, cfg.segment_num, cfg.update_frac, cnt))

        torch.save(ord, "%s%s_crf%d_F%d_seg%d_frac%.2f_epoch%d_batch%d.pth" % \
            (cfg.save_path, video_basename, cfg.crf, cfg.F, cfg.segment_length, cfg.update_frac, cfg.epoch, cfg.batch_size))
        print("Saved the SRVC model for %s video file" % (video_basename))
        lr_cap.release()
        hr_cap.release()

if __name__ == "__main__":
    train()