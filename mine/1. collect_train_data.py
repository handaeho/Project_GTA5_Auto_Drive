import numpy as np
from mine.grabscreen import grab_screen
import cv2
import time
from out.getkeys import key_check
import os


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
    nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output


starting_value = 1
while True:
    # 저장할 경로 알아서 변경
    file_name = f'C:/dev/project/gta5/training_data-{starting_value}.npy'

    if os.path.isfile(file_name):
        print('File exists, moving along', starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!', starting_value)

        break


def main(filename, start_value):
    for i in range(4, 0, -1):
        print(i)
        time.sleep(1)
    print('STARTING!!!')

    paused = False
    last_time = time.time()
    training_data = []
    while True:
        if not paused:
            # 해상도 알아서
            width = 1280
            height = 720
            screen = grab_screen(region=(0, 25, width, height+25))
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (480, 270))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen, output])

            # print(f'loop took {time.time() - last_time} seconds')
            last_time = time.time()

            # 이미지그랩확인, q 누르면 종료
            # 확인만 하고 91행까지 주석처리
            cv2.imshow('window', screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            if len(training_data) % 100 == 0:
                print(len(training_data))

            if len(training_data) == 1000:
                np.save(filename, training_data)
                print('SAVED')
                training_data = []
                start_value += 1
                # 저장할 경로(file_name과 같음)
                filename = f'C:/dev/project/gta5/training_data-{start_value}.npy'

        keys = key_check()

        # T를 누르면 일시정지/다시시작
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


if __name__ == '__main__':
    main(file_name, starting_value)
