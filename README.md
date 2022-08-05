# SRVC - PyTorch Implementation

Exp1:
crf 10, segment_length 5, F 32 as default
Calculate average PSNR, SSIM, bits per pixel of all 11 videos

Exp2:
segment_length 5, F 32 as default
crf - 10, 25, 35, 50
Calculate average PSNR, SSIM, bits per pixel of 1 video (166010169)

Exp3: crf10, segment_length 5 as default
F - 8, 16, 32, 64, 128
Calculate average PSNR, SSIM, Inference Time, Number of Parameters of 1 video (166010169)

Exp4: crf10, F 32 as default
segment_length - 5, 10, 15, 20, inf
Calculate average PSNR, SSIM, bits per pixel of 1 video (166010169)