
import time
import mine.models
import numpy as np

model = mine.models.inception3((270, 480, 3))
WEIGHT = 'C:/dev/projectGTA/gta5/model_weight.h5'
model.load_weights(WEIGHT)
print('가중치 불러오기 완료')

file_name = f'C:/dev/projectGTA/gta5/training_data-{101}.npy'
train_data = np.load(file_name, allow_pickle=True)
X = np.array([i[0] for i in train_data])
Y = np.array([i[1] for i in train_data])

last_time = time.time()
for x in X:
    screen = x[np.newaxis, :]
    pred = model.predict(screen)
    print(pred)
    print(time.time() - last_time)
    last_time = time.time()