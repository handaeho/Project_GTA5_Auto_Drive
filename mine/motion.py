# https://blog.mcv.kr/12

import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import time


def scan(pix, pix2):
    # 프레임간의 픽셀차이를 계산
    # 0에 가까울수록 일치, 255에 가까울수록 불일치
    a = []
    b = []
    for y in range(0, 480):
        for x in range(0, 270):
            a.append(np.int((abs(np.int(pix[x][y][0]) - np.int(pix2[x][y][0])) +
                            abs(np.int(pix[x][y][1]) - np.int(pix2[x][y][1])) +
                            abs(np.int(pix[x][y][2]) - np.int(pix2[x][y][2]))) / 3))
        b.append(sum(a, 0.0) / len(a))
        a = []
    return (sum(b, 0.0) / len(b))


if __name__ == '__main__':

    file_name = f'C:/dev/projectGTA/gta5/training_data-{101}.npy'
    data = np.load(file_name, allow_pickle=True)
    t0 = data[0, 0]
    t1 = data[1, 0]
    t2 = data[2, 0]
    imglist = data[:, 0]
    label = data[:, 1]
    print(label)

    prev = imglist[0]
    # last = time.time()
    for idx, img in enumerate(imglist):
        now = img
        print(f'{idx+1}번째 일치도 {scan(prev, now)}')
        # print(time.time() - last)
        # last = time.time()
        prev = img

