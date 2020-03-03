from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.xception import Xception

def inception3(input_shape: tuple):
    # 기준모델
    base_model = InceptionV3(weights=None, include_top=False, input_shape=input_shape)
    # 글로벌 공간 평균값 풀링 레이어를 더합니다
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # 완전 연결 레이어를 더합니다
    x = Dense(1024, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    predictions = Dense(9, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


def xception(input_shape):
    base_model = Xception(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(9, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


if __name__ == '__main__':
    model = xception((270, 480, 3))
    model.summary()