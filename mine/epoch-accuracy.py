import mine.models
import matplotlib.pyplot as plt
import numpy as np

file_name = f'C:/dev/projectGTA/gta5/training_data-{124}.npy'
data = np.load(file_name, allow_pickle=True)
X_test = np.array([i[0] for i in data])
Y_test = np.array([i[1] for i in data])

for i in range(1, 17):
    model = mine.models.xception((270, 480, 3))
    WEIGHT = f'C:/dev/projectGTA/gta5/model_weight_euro_{i}epoch.h5'
    model.load_weights(WEIGHT)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    pred = model.predict(X_test)
    print(pred)
