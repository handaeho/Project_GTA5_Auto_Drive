import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import collections

num = 71

file_name = f'C:/dev/projectGTA/gta5/training_data-{num}.npy'
train_data = np.load(file_name, allow_pickle=True)

pick = []
for idx, i in enumerate(train_data[:, 1]):
    # print(i)
    mode_choice = np.argmax(i)
    print(mode_choice)
    pick.append(mode_choice)

a = collections.Counter(pick)
print(a)

for idx, i in enumerate(train_data[:, 0]):
    i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(i)
    img.save(f'C:/dev/projectGTA/gta5/image/{num}-{idx}.jpg')

