"""

"""
import cv2 as cv
import os
import time
import numpy as np

def image_io_demo():
    for filename in ['car.jpeg', 'eve.png']:
        image = cv.imread(filename)
        h, w, c = image.shape
        print(filename)
        print(h, w, c)
        base_name, ext_name = os.path.splitext(filename)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        print('gray.shape:', gray.shape)
        gray_name = base_name + '-gray.' + ext_name
        cv.imwrite(gray_name, gray)
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        print('hsv.shape:', hsv.shape)
        hsv_name = base_name + '-hsv.' + ext_name
        cv.imwrite(hsv_name, hsv)

def video_io_demo():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print('no working camera')
            return
        print('开始捕捉视频图片')
        cap_time = time.time()
        name = "cap-" + str(cap_time)
        cv.imwrite(name + '.png', frame)
        print('生成捕捉图片：', name)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_name = name + '-gray'
        cv.imwrite(gray_name + '.png', gray)
        print('生成捕捉图片：', gray_name)
        next = input("输入q回车退出")
        if next == 'q':
            break
    cap.release()

def basic_ops_demo():
    filename = 'car.jpeg'
    base_name, ext_name = os.path.splitext(filename)
    image = cv.imread(filename)
    h, w, c = image.shape
    print(h, w, c)
    mv = cv.split(image)
    red_name = base_name + '-red.' + ext_name
    green_name = base_name + '-green.' + ext_name
    blue_name = base_name + '-blue.' + ext_name
    cv.imwrite(red_name, mv[0])
    cv.imwrite(green_name, mv[1])
    cv.imwrite(blue_name, mv[2])
    big_name = base_name + '-big.' + ext_name
    big = cv.resize(image, (600, 600))
    cv.imwrite(big_name, big)
    small_name = base_name + '-small.' + ext_name
    small = cv.resize(image, (30, 30))
    cv.imwrite(small_name, small)
    print(image.shape)
    image_blob = image.transpose(2, 0, 1)
    print(image_blob.shape)
    image_blob = np.expand_dims(image_blob, 0)
    print(image_blob.shape)

    a = np.array([1, 2, 3, 4, 5, 6, 9, 88, 0, 12, 14, 5, 6])
    index = np.argmax(a)
    print(index)
    print(a[index])

    for row in range(h):
        for col in range(w):
            b, g, r = image[row, col]
            print(b, g, r)



if __name__ == "__main__":
    #image_io_demo()
    video_io_demo()
    #basic_ops_de mo()