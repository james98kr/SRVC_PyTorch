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
            frame = torch.unsqueeze(torch.from_numpy(frame), 0)
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
            frame = torch.unsqueeze(torch.from_numpy(frame), 0)
            frames.append(frame)
        return frames


def YUV2RGB(yuv):
    m = np.array([[1.0, 1.0, 1.0],
                [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])
    h = int(yuv.shape[0] / 1.5)
    w = yuv.shape[1]
    y = yuv[:h]
    h_u = h // 4
    h_v = h // 4
    u = yuv[h:h + h_u]
    v = yuv[-h_v:]
    u = np.reshape(u, (h_u * 2, w // 2))
    v = np.reshape(v, (h_v * 2, w // 2))
    u = cv2.resize(u, (w, h), interpolation=cv2.INTER_CUBIC)
    v = cv2.resize(v, (w, h), interpolation=cv2.INTER_CUBIC)
    yuv = np.concatenate([y[..., None], u[..., None], v[..., None]], axis=-1)

    bgr = np.dot(yuv, m)
    bgr[:, :, 0] -= 179.45477266423404
    bgr[:, :, 1] += 135.45870971679688
    bgr[:, :, 2] -= 226.8183044444304
    bgr = np.clip(bgr, 0, 255)

    return bgr.astype(np.uint8)


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
        rgb = YUV2RGB(yuv)
        return ret, rgb

    def isOpened(self):
        return self.is_opened

    def release(self):
        try:
            self.f.close()
        except Exception as e:
            print(str(e))