import cv2
import demo

if __name__ == '__main__':
    img1 = cv2.imread('img1.jpg')
    demo.NumpyUint83CToCvMat(img1)
    img3 = demo.CvMatUint83CToNumpy()
    cv2.imwrite('img3.jpg', img3)

