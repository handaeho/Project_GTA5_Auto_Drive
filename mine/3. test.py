import numpy as np
from mine.grabscreen import grab_screen
import cv2
import time
from mine.directkeys import PressKey, ReleaseKey, W, A, S, D
from mine.getkeys import key_check
import random
import numpy as np
import mine.models
from mine.motion import scan

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# 텐서 2.1에서 에러 안나게
# 1.x 버전대는 코드 다름
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# 본인 게임 해상도
GAME_WIDTH = 1280
GAME_HEIGHT = 720

WIDTH = 480
HEIGHT = 270
EPOCHS = 10

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]



def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def left():
    ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def right():
    ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)


def no_keys():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


model = mine.models.inception3((270, 480, 3))
# 가중치 저장된 경로
WEIGHT = 'C:/dev/projectGTA/gta5/model_weight.h5'
model.load_weights(WEIGHT)
print('가중치 불러오기 완료')


def main():
    last_time = time.time()
    for i in range(4, 0, -1):
        print(i)
        time.sleep(1)

    paused = False
    mode_choice = 0

    screen = grab_screen(region=(0, 25, GAME_WIDTH, GAME_HEIGHT+25))
    # prev = cv2.resize(screen, (WIDTH, HEIGHT))
    similarity = 0
    now = screen

    while True:
        if not paused:
            prev = now
            screen = grab_screen(region=(0, 25, GAME_WIDTH, GAME_HEIGHT+25))
            now = screen
            # print(screen.shape)
            last_time = time.time()
            screen = cv2.resize(screen, (WIDTH, HEIGHT))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            # print(screen.shape)
            screen = screen[np.newaxis, :]
            # print(screen.shape)
            prediction = model.predict(screen)
            # print(prediction.shape)
            print(prediction)
            mode_choice = np.argmax(prediction)
            # print(mode_choice)

            if mode_choice == 0:
                straight()
                choice_picked = 'straight'

            elif mode_choice == 1:
                reverse()
                choice_picked = 'reverse'

            elif mode_choice == 2:
                left()
                choice_picked = 'left'
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
            elif mode_choice == 4:
                forward_left()
                choice_picked = 'forward+left'
            elif mode_choice == 5:
                forward_right()
                choice_picked = 'forward+right'
            elif mode_choice == 6:
                reverse_left()
                choice_picked = 'reverse+left'
            elif mode_choice == 7:
                reverse_right()
                choice_picked = 'reverse+right'
            elif mode_choice == 8:
                no_keys()
                choice_picked = 'nokeys'

            print(choice_picked)

            # 209행까지 비슷한 프레임이 반복되면 빠져나오는 코드
            # 이걸 주석처리 안할시 loop 한번당 연산속도가 약 0.5초 이상 더 걸리는 걸로 보임
            diff = scan(prev, now)
            print(diff)
            if diff < 10:
                similarity += 1

            if similarity > 5:
                solution = random.randrange(0, 4)
                if solution == 0:
                    reverse()
                    time.sleep(random.uniform(1, 2))
                    forward_left()
                    time.sleep(random.uniform(1, 2))

                elif solution == 1:
                    reverse()
                    time.sleep(random.uniform(1, 2))
                    forward_right()
                    time.sleep(random.uniform(1, 2))

                elif solution == 2:
                    reverse_left()
                    time.sleep(random.uniform(1, 2))
                    forward_right()
                    time.sleep(random.uniform(1, 2))

                elif solution == 3:
                    reverse_right()
                    time.sleep(random.uniform(1, 2))
                    forward_left()
                    time.sleep(random.uniform(1, 2))
                similarity = 0

        keys = key_check()

        # T를 눌러서 일시정지/해제
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)


main()
