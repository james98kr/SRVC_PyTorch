import cv2
import numpy as np
import torch

def get_segment_frames(cap, frame_per_segment):
    cap_type = 'cv2_cap' if isinstance(cap, cv2.VideoCapture) else 'yuv_cap'
    frames = []
    assert cap.isOpened()

    if cap_type == 'yuv_cap':
        for n in range(frame_per_segment):
            ret, frame = cap.read()
            if not ret:
                break
                # raise Exception("Unable to read frame!")
            frame = np.transpose(frame, (2,0,1))
            frame = torch.from_numpy(frame)
            frames.append(frame)
        return frames
    else:
        for n in range(frame_per_segment):
            ret, frame = cap.read()
            if not ret:
                break
                # raise Exception("Unable to read frame!")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2,0,1))
            frame = torch.from_numpy(frame)
            frames.append(frame)
        return frames

class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.width, self.height = size
        self.frame_len = (self.width * self.height * 3) // 2
        self.f = open(filename, 'rb')
        self.is_opened = True
        self.shape = (int(self.height * 1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            self.is_opened = False
            return False, None
        return True, yuv

    def read(self):
        # Note that output has shape (H,W,C), which must be transposed
        # to (C,H,W) in order to be processed by the model
        ret, yuv = self.read_raw()
        if not ret:
            return ret, None
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
        return ret, rgb

    def isOpened(self):
        return self.is_opened

    def release(self):
        try:
            self.f.close()
        except Exception as e:
            print(str(e))