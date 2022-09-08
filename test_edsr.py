from utils.video_reader import *
from utils.functions import *
from model.sr_model import *
from model.edsr import *
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import cv2
import os
import time


edsr_args = {'n_resblocks': 16,
            'n_feats': 64,
            'scale': [4],
            'rgb_range': 255,
            'n_colors': 3,
            'res_scale': 1}


def test():
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')

    # Get all configurations and set up model
    cfg = get_configs()
    edsr = EDSR(edsr_args)
    edsr.to(device)
    edsr.eval()

    # Bicubic upsampler for comparison
    bicubic_upsampler = nn.Upsample(scale_factor=cfg.scale, mode='bicubic').to(device)

    # Functions for calculating L2Loss and PSNR
    criterion = nn.MSELoss() if not cfg.L1_loss else nn.L1Loss()
    psnr_calculate = PeakSignalNoiseRatio().to(device)
    ssim_calculate = StructuralSimilarityIndexMeasure().to(device)

    for saved_file in cfg.saved_file_list:
        video_basename = os.path.basename(saved_file).split('_')
        video_basename = video_basename[1] + '_' + video_basename[2]
        lr_video = cfg.lr_path + video_basename + '_crf' + str(cfg.crf) + '_dec.yuv'
        hr_video = cfg.hr_path + video_basename + '.yuv'

        ######### In order to exclude some files from the testing process, insert their basename here #########
        if video_basename not in ['vimeo_204439769']:
            continue
        ######### In order to exclude some files from the testing process, insert their basename here #########
        print(saved_file)
        print(lr_video)
        print(hr_video)


        # Open log file for record
        log = open("%slog_%s_crf%d_F%d_seg%d.txt" % (cfg.log_path, video_basename, cfg.crf, cfg.F, cfg.segment_length), 'w')

        lr_cap = VideoCaptureYUV(lr_video, cfg.lr_size)
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
            # Load input and output frames for current segment
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
            # Load saved parameters appropriately
            if segment == cfg.segment_skip or cfg.update_frac == 1:
                edsr.load_state_dict(saved_parameters[segment])
            else:
                i = edsr.state_dict()
                p = {v: i[v].clone() for v in i.keys()}
                segment_param = saved_parameters[segment]
                for v in p.keys():
                    if len(segment_param[v]) == 0:
                        continue
                    delta, coords = list(zip(*segment_param[v]))
                    if isinstance(coords[0], tuple):
                        a, b, c, d = list(zip(*coords))
                        p[v][list(a), list(b), list(c), list(d)] += torch.Tensor(list(delta)).to(device)
                    elif isinstance(coords[0], int):
                        p[v][list(coords)] += torch.Tensor(list(delta)).to(device)
                    else:
                        raise Exception("A huge error!")
                edsr.load_state_dict(p)
            
            for i, (input_frame, output_frame) in enumerate(zip(input_frames, output_frames)):
                input_frame = input_frame.to(torch.float32).to(device) / 255
                output_frame = output_frame.to(torch.float32).to(device) / 255
                t0 = time.time()
                my_output_frame = edsr(input_frame).to(torch.float32)
                t1 = time.time()
                my_output_frame = threshold_output(my_output_frame)
                loss = criterion(my_output_frame, output_frame).cpu().detach().numpy()

                output_frame = (output_frame * 255).to(torch.uint8)
                my_output_frame = (my_output_frame * 255).to(torch.uint8)
                psnr = psnr_calculate(my_output_frame, output_frame).cpu().detach().numpy()
                output_frame = output_frame.to(torch.float32)
                my_output_frame = my_output_frame.to(torch.float32)
                ssim = ssim_calculate(my_output_frame, output_frame).cpu().detach().numpy()

                bicubic_output_frame = bicubic_upsampler(input_frame).to(torch.float32)
                output_frame = output_frame.to(torch.uint8)
                bicubic_output_frame = (bicubic_output_frame * 255).to(torch.uint8)
                bicubic_psnr = psnr_calculate(bicubic_output_frame, output_frame).cpu().detach().numpy()
                output_frame = output_frame.to(torch.float32)
                bicubic_output_frame = bicubic_output_frame.to(torch.float32)
                bicubic_ssim = ssim_calculate(bicubic_output_frame, output_frame).cpu().detach().numpy()

                if segment == 10 and i == 5:
                    torch.save((input_frame * 255).to(torch.uint8), "./inputframe.pt")
                    torch.save(output_frame, "./outputframe.pt")
                    torch.save(my_output_frame, "./myoutputframe.pt")
                    torch.save(bicubic_output_frame, "./bicubicoutputframe.pt")

                total_time += (t1 - t0)
                if (psnr != -np.inf):    
                    total_psnr += psnr
                else:
                    total_psnr += 30
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