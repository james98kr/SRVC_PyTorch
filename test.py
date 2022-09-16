from utils.video_reader import *
from utils.functions import *
from utils.dataset import *
from model.sr_model import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from collections import OrderedDict
import os
import time

def test():
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')

    # Get all configurations and set up model
    cfg = get_configs()
    sr_model = nn.DataParallel(SR_Model(cfg.F, cfg.scale, cfg.patch_size))
    sr_model.to(device)
    sr_model.eval()

    # Bicubic upsampler for comparison
    bicubic_upsampler = nn.Upsample(scale_factor=cfg.scale, mode='bicubic').to(device)

    # Functions for calculating L2Loss and PSNR
    criterion = nn.MSELoss() if not cfg.L1_loss else nn.L1Loss()
    psnr_calculate = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_calculate = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    for saved_file in cfg.saved_file_list:
        print(saved_file)
        video_basename = os.path.basename(saved_file).split('_')
        video_basename = video_basename[0] + '_' + video_basename[1]
        lr_video = cfg.lr_path + video_basename + '_crf' + str(cfg.crf) + '.mp4'
        hr_video = cfg.hr_path + video_basename + '.yuv'

        ######### In order to exclude some files from the testing process, insert their basename here #########
        if video_basename in []:
            continue
        ######### In order to exclude some files from the testing process, insert their basename here #########
    
        # Open log file for record
        log = open("%slog_%s_crf%d_F%d_seg%d_frac%.2f_epoch%d_batch%d.txt" % \
            (cfg.log_path, video_basename, cfg.crf, cfg.F, cfg.segment_length, cfg.update_frac, cfg.epoch, cfg.batch_size), 'w')

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
            before = sr_model.state_dict()
            before = {v: before[v].clone() for v in before.keys()}
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
                sr_model.load_state_dict(saved_parameters[segment])
            else:
                p = sr_model.state_dict()
                p = OrderedDict({v: p[v].clone() for v in p.keys()})
                segment_param = saved_parameters[segment]
                for key in segment_param.keys():
                    actual = segment_param[key][0]
                    coords = segment_param[key][1]
                    if isinstance(coords, tuple):
                        for i in range(len(actual)):
                            p[key][int(coords[0][i]),int(coords[1][i]),int(coords[2][i]),int(coords[3][i])] += actual[i]
                    else:
                        for i in range(len(actual)):
                            p[key][int(coords[i])] += actual[i]
                sr_model.load_state_dict(p)

            # For checking how many parameters are actually updated
            after = sr_model.state_dict()
            after = {v: after[v].clone() for v in after.keys()}
            delta = {v: (after[v] - before[v]) for v in after.keys()}
            cnt = 0
            for v in delta.keys():
                cnt += torch.count_nonzero(delta[v])
            print("segment: %d, cnt: %d" % (segment + 1, cnt))

            # Actual inference on full frame
            srvc_dataset = SRVC_DataSet(input_frames, output_frames)
            srvc_dataloader = DataLoader(srvc_dataset, batch_size=1, shuffle=False)
            for i, (input_frame, output_frame) in enumerate(srvc_dataloader):
                input_frame = input_frame.to(torch.float32).to(device) / 255
                output_frame = output_frame.to(torch.float32).to(device) / 255
                t0 = time.time()
                my_output_frame = sr_model(input_frame).to(torch.float32)
                t1 = time.time()
                my_output_frame = threshold_output(my_output_frame).to(torch.float32)
                loss = criterion(my_output_frame, output_frame).cpu().detach().numpy()
                psnr = psnr_calculate(my_output_frame, output_frame).cpu().detach().numpy()
                ssim = ssim_calculate(my_output_frame, output_frame).cpu().detach().numpy()

                bicubic_output_frame = bicubic_upsampler(input_frame).to(torch.float32)
                bicubic_psnr = psnr_calculate(bicubic_output_frame, output_frame).cpu().detach().numpy()
                bicubic_ssim = ssim_calculate(bicubic_output_frame, output_frame).cpu().detach().numpy()

                total_time += (t1 - t0)
                if psnr != -np.inf and psnr != np.inf:    
                    total_psnr += psnr
                if ssim != -np.inf and ssim != np.inf:
                    total_ssim += ssim
                if bicubic_psnr != -np.inf and bicubic_psnr != np.inf:
                    total_bicubic_psnr += bicubic_psnr
                if bicubic_ssim != -np.inf and bicubic_ssim != np.inf:
                    total_bicubic_ssim += bicubic_ssim
                total_frames += 1
                if i % 10 == 0:
                    print("Video: %s, Segment: %d/%d, Frame: %d/%d, L2Loss: %f, PSNR: %f, SSIM: %f, Bicubic PSNR: %f, Bicubic SSIM: %f, Inference Time: %f" \
                        % (video_basename, segment + 1, cfg.segment_num, i, len(srvc_dataloader), loss, psnr, ssim, bicubic_psnr, bicubic_ssim, t1 - t0))
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