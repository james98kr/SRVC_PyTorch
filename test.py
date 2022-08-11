from utils.video_reader import *
from utils.functions import *
from model.sr_model import *
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import cv2
import os
import time


def test():
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')

    # Get all configurations and set up model
    cfg = get_configs()
    sr_model = SR_Model(cfg.F, cfg.scale, cfg.patch_size)
    sr_model.to(device)
    sr_model.eval()

    # Bicubic upsampler for comparison
    bicubic_upsampler = nn.Upsample(scale_factor=cfg.scale, mode='bicubic').to(device)

    # Functions for calculating L2Loss and PSNR
    criterion = nn.MSELoss() if not cfg.L1_loss else nn.L1Loss()
    psnr_calculate = PeakSignalNoiseRatio().to(device)
    ssim_calculate = StructuralSimilarityIndexMeasure().to(device)

    for saved_file in cfg.saved_file_list:
        video_basename = os.path.basename(saved_file).split('_')
        video_basename = video_basename[0] + '_' + video_basename[1]
        lr_video = cfg.lr_path + video_basename + '_crf' + str(cfg.crf) + '.mp4'
        hr_video = cfg.hr_path + video_basename + '.yuv'

        ######### In order to exclude some files from the testing process, insert their basename here #########
        if video_basename in []:
            continue
        ######### In order to exclude some files from the testing process, insert their basename here #########

        # Open log file for record
        log = open("%slog_%s_crf%d_F%d_seg%d.txt" % (cfg.log_path, video_basename, cfg.crf, cfg.F, cfg.segment_length), 'w')

        lr_cap = cv2.VideoCapture(lr_video)
        hr_cap = VideoCaptureYUV(hr_video, cfg.hr_size)
        fps = find_video_fps(video_basename, cfg.original_path)
        saved_parameters = torch.load(saved_file)
        total_time = 0
        total_psnr = 0
        total_bicubic_psnr = 0
        total_ssim = 0
        total_bicubic_ssim = 0
        total_frames = 0
    
        for segment in range(cfg.segment_num):
            # Load saved parameters appropriately
            if segment == 0:
                sr_model.load_state_dict(saved_parameters[segment])
            else:
                i = sr_model.state_dict()
                p = {v: i[v].clone() for v in i.keys()}
                segment_param = saved_parameters[segment]
                for v in p.keys():
                    if len(segment_param[v]) == 0:
                        continue
                    delta, coords = list(zip(*segment_param[v]))
                    if isinstance(coords[0], tuple):
                        a, b, c, d = list(zip(*coords))
                        p[v][a, b, c, d] += torch.Tensor(list(delta)).to(device)
                    elif isinstance(coords[0], int):
                        p[v][list(coords)] += torch.Tensor(list(delta)).to(device)
                    else:
                        raise Exception("A huge error!")
                sr_model.load_state_dict(p)

            # Load input and output frames for current segment
            input_frames = get_segment_frames(lr_cap, cfg.segment_length * fps)
            output_frames = get_segment_frames(hr_cap, cfg.segment_length * fps)
            assert len(input_frames) == len(output_frames)
            
            for i, (input_frame, output_frame) in enumerate(zip(input_frames, output_frames)):
                input_frame = input_frame.to(torch.float32).to(device) / 127.5 - 1.0
                output_frame = output_frame.to(torch.float32).to(device) / 127.5 - 1.0
                t0 = time.time()
                my_output_frame = sr_model(input_frame).to(torch.float32)
                t1 = time.time()
                loss = criterion(my_output_frame, output_frame).cpu().detach().numpy()
                psnr = psnr_calculate(my_output_frame, output_frame).cpu().detach().numpy()
                ssim = ssim_calculate(my_output_frame, output_frame).cpu().detach().numpy()

                bicubic_output_frame = bicubic_upsampler(input_frame).to(torch.float32)
                bicubic_psnr = psnr_calculate(bicubic_output_frame, output_frame).cpu().detach().numpy()
                bicubic_ssim = ssim_calculate(bicubic_output_frame, output_frame).cpu().detach().numpy()

                total_time += (t1 - t0)
                total_psnr += psnr
                total_ssim += ssim
                total_bicubic_psnr += bicubic_psnr
                total_bicubic_ssim += bicubic_ssim
                total_frames += 1
                if i % 10 == 0:
                    print("Video: %s, Segment: %d/%d, Frame: %d/%d, L2Loss: %f, PSNR: %f, SSIM: %f, Bicubic PSNR: %f, Bicubic SSIM: %f, Inference Time: %f" \
                        % (video_basename, segment + 1, cfg.segment_num, i, len(input_frames), loss, psnr, ssim, bicubic_psnr, bicubic_ssim, t1 - t0))
                
                ssim_calculate.reset()

        # When all the frames in each video has gone through inference, calculate average PSNR and average SSIM
        average_inference_time = total_time / total_frames
        average_psnr = total_psnr / total_frames
        average_ssim = total_ssim / total_frames
        average_bicubic_psnr = total_bicubic_psnr / total_frames
        average_bicubic_ssim = total_bicubic_ssim / total_frames
        print("Final Result for Video %s:" % (video_basename))
        print("Video: %s, Duration: %d, Segment length: %d, Frames per segment: %d, Average Inference Time: %f, Average PSNR: %f, Average SSIM: %f, Average Bicubic PSNR: %f, Average Bicubic SSIM: %f" \
            % (video_basename, cfg.end_time - cfg.start_time, cfg.segment_length, fps * cfg.segment_length, average_inference_time, average_psnr, average_ssim, average_bicubic_psnr, average_bicubic_ssim))
        log.write("Video: %s, \nDuration: %d, \nSegment length: %d, \nFrames per segment: %d, \nAverage Inference Time: %f, \nAverage PSNR: %f, \nAverage SSIM: %f, \nAverage Bicubic PSNR: %f, \nAverage Bicubic SSIM: %f" 
                % (video_basename, cfg.end_time - cfg.start_time, cfg.segment_length, fps * cfg.segment_length, average_inference_time, average_psnr, average_ssim, average_bicubic_psnr, average_bicubic_ssim))
        log.close()


if __name__ == "__main__":
    test()