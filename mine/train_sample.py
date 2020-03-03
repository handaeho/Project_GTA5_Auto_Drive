import numpy as np
import random

FILE_NUM = 10  # npy 파일 갯수
LR = 1e-3
EPOCHS = 30
# MODEL_NAME = 'inception_v3'
# PREV_MODEL = 'inception'
# LOAD_MODEL = True

# model = googlenet(WIDTH, HEIGHT, LR, output=9)

# if LOAD_MODEL:
#     model.load(PREV_MODEL)
#     print('We have loaded a previous model!!!!')

start = 1
file_name = f'C:/dev/project/gta5/training_data-{start}.npy'
train_data = np.load(file_name, allow_pickle=True)
for i in range(start + 1, start + 10):
    file_name = f'C:/dev/project/gta5/training_data-{i}.npy'
    data = np.load(file_name, allow_pickle=True)
    train_data = np.append(train_data, data, axis=0)
print(train_data.shape)

random.shuffle(train_data)
cut = int(0.8 * len(train_data))
train = train_data[:cut]
validation = train_data[cut:]
print(train[0, 0].shape)

X = np.array([i[0] for i in train])
Y = np.array([i[1] for i in train])
print(X.shape)
print(Y.shape)
X_val = np.array([i[0] for i in validation])
Y_val = np.array([i[1] for i in validation])

# model.fit({'input': X}, {'targets': Y},
#           n_epoch=1, validation_set=({'input': X_val}, {'targets': Y_val}),
#           snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)
#
# model.save(MODEL_NAME)

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# 선행학습된 기준모델을 만듭니다
base_model = InceptionV3(weights=None, include_top=False, input_shape=(270, 480, 3))

# 글로벌 공간 평균값 풀링 레이어를 더합니다
x = base_model.output
x = GlobalAveragePooling2D()(x)
# 완전 연결 레이어를 더합니다
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)

# 로지스틱 레이어를 더합니다 -- 9가지 클래스
predictions = Dense(9, activation='softmax')(x)

# 다음은 학습할 모델입니다
model = Model(inputs=base_model.input, outputs=predictions)

# 첫째로: (난수로 초기값이 설정된) 가장 상위 레이어들만 학습시킵니다
# 다시 말해서 모든 InceptionV3 콘볼루션 레이어를 고정합니다
for layer in base_model.layers:
    layer.trainable = False

# 모델을 컴파일합니다 (*꼭* 레이어를 학습불가 상태로 세팅하고난 *후*에 컴파일합니다)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()

# 모델을 새로운 데이터에 대해 몇 세대간 학습합니다
model.fit(X, Y, validation_data=(X_val, Y_val))

# 이 시점에서 상위 레이어들은 충분히 학습이 되었기에,
# inception V3의 콘볼루션 레이어에 대한 파인튜닝을 시작합니다
# 가장 밑 N개의 레이어를 고정하고 나머지 상위 레이어를 학습시킵니다

# 레이어 이름과 레이어 인덱스를 시각화하여
# 얼마나 많은 레이어를 고정시켜야 하는지 확인합니다:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# 가장 상위 2개의 inception 블록을 학습하기로 고릅니다,
# 다시 말하면 첫 249개의 레이어는 고정시키고 나머지는 고정하지 않습니다:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# 이러한 수정사항이 효과를 내려면 모델을 다시 컴파일해야 합니다
# 낮은 학습 속도로 세팅된 SGD를 사용합니다
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# 다시 한 번 모델을 학습시킵니다
# (이번엔 상위 2개의 inception 블록을 상위의 밀집 레이어들과 함께 파인튜닝합니다)
model.fit(X, Y, validation_data=(X_val, Y_val))

model.save('C:/dev/project/gta5/model_weight')