import mine.models
import keras
import numpy as np
import random

model = keras.models.load_model('C:/dev/project/gta5/model_weight')
# model.summary()

start = 5
file_name = f'C:/dev/project/gta5/training_data-{start}.npy'
test_data = np.load(file_name, allow_pickle=True)
test_data = test_data[0, 0]
test_data = test_data[np.newaxis, :]
print(test_data.shape)
pred = model.predict(test_data)
print(pred.shape)
print(pred)
# for i in range(start + 1, start + 5):
#     file_name = f'C:/dev/project/gta5/training_data-{i}.npy'
#     data = np.load(file_name, allow_pickle=True)
#     train_data = np.append(train_data, data, axis=0)
# print(train_data.shape)
#
# random.shuffle(train_data)
# cut = int(0.8 * len(train_data))
# train = train_data[:cut]
# validation = train_data[cut:]
# print(train[0, 0].shape)
#
# X = np.array([i[0] for i in train])
# Y = np.array([i[1] for i in train])
# print(X.shape)
# print(Y.shape)
# X_val = np.array([i[0] for i in validation])
# Y_val = np.array([i[1] for i in validation])
#
# model.fit(X, Y, batch_size=10, epochs=1, validation_data=(X_val, Y_val))
# model.save('C:/dev/project/gta5/model_weight')