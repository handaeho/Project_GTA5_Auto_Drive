import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg



def scan(pix, pix2):
    a = []
    b = []
    for y in range(0, 480):
        for x in range(0, 270):
            a.append(int((abs(int(pix[x][y][0]) - int(pix2[x][y][0])) +
                                   abs(int(pix[x][y][1]) - int(pix2[x][y][1])) +
                                   abs(int(pix[x][y][2]) - int(pix2[x][y][2]))) / 3))
        b.append(sum(a, 0.0) / len(a))
        a = []
    return (sum(b, 0.0) / len(b))


if __name__ == '__main__':

    file_name = f'C:/dev/project/gta5/training_data-{5}.npy'
    data = np.load(file_name, allow_pickle=True)
    t0 = data[0, 0]
    t1 = data[1, 0]
    t2 = data[2, 0]
    imglist = data[:, 0]
    label = data[:, 1]
    print(label)

    prev = imglist[0]
    for idx, img in enumerate(imglist):
        now = img
        print(f'{idx+1}번째 일치도 {scan(prev, now)}')
        prev = img


