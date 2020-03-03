# https://www.programcreek.com/python/example/14102/win32gui.GetDesktopWindow Example4
"""
화면 캡쳐
PIL.ImageGrab이나 wxPython을 사용하는 것보다
속도가 월등히 빨라 높은 FPS를 얻을 수 있음
"""
import cv2
import win32gui, win32ui, win32con, win32api
import numpy as np


def grab_screen(region=None):
    """

    :param region: 게임 화면의 위치 (x1, y1, x2, y2)를 입력
    :return: 이미지 배열
    """
    hwin = win32gui.GetDesktopWindow()

    if region:
        x1, y1, x2, y2 = region
        width = x2 - x1 + 1
        height = y2 - y1 + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        x1 = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        y1 = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (x1, y1), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


if __name__ == '__main__':
    while True:
        screen = grab_screen()
        print(screen.shape)
        screen = cv2.resize(screen, (480, 270))
        print(screen.shape)
        # run a color convert:
        # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        cv2.imshow('window', cv2.resize(screen, (640, 360)))
        print(screen.shape)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
