import cv2
import numpy as np
import matplotlib.pyplot as plt

# 二值化处理
def myThreshold(img, threshold):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < threshold:
                img[i, j] = 0
            elif img[i, j] > threshold:
                img[i, j] = 255
    return img

# robert算子[[-1,-1],[1,1]]
def myRobert(img):
    r, c = img.shape
    r_sunnzi = [[-1,-1], [1,1]]
    for x in range(r):
        for y in range(c):
            if (y + 2 <= c) and (x + 2 <= r):
                imgChild = img[x:x+2, y:y+2]
                list_robert = r_sunnzi * imgChild
                img[x, y] = abs(list_robert.sum()) # 求和加绝对值
    return img
                 
# sobel算子
def mySobel(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1,0,1], [-2,0,2], [-1,0,1]]) # X方向
    s_suanziY = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])     
    for i in range(r-2):
        for j in range(c-2):
            new_imageX[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziX))
            new_imageY[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziY))
            new_image[i+1, j+1] = (new_imageX[i+1, j+1] * new_imageX[i+1, j+1] + new_imageY[i+1, j+1] * new_imageY[i+1, j+1]) ** 0.5
    # return np.uint8(new_imageX)
    # return np.uint8(new_imageY)
    return np.uint8(new_image) # 无方向算子处理的图像
 
# Laplace算子[[0,1,0],[1,-4,1],[0,1,0]]或[[1,1,1],[1,-8,1],[1,1,1]]
def myLaplace(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    L_sunnzi = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])     
    # L_sunnzi = np.array([[1,1,1], [1,-8,1], [1,1,1]])      
    for i in range(r-2):
        for j in range(c-2):
            new_image[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * L_sunnzi))
    return np.uint8(new_image)
 
 
img_path = 'E:\\image.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转化为灰度图
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
'''
#! 伽马变换
fi = img / 255.0 # 8位图归一化处理（将灰度图像素值调整到[0,1]之间）
gamma = 1.0 # 0 < gamma < 1降低对比度 gamma > 1增加对比度
img_gamma = np.power(fi, gamma) # 对图像矩阵中的每一个值进行幂运算
cv2.namedWindow('img_gamma', cv2.WINDOW_NORMAL)
cv2.imshow('img_gamma', img_gamma)
'''
'''
#! 全局直方图均衡化
img_equa = cv2.equalizeHist(img)
cv2.namedWindow('img_equa', cv2.WINDOW_NORMAL)
cv2.imshow('img_equa', img_equa)
# 绘制直方图
hist = cv2.calcHist([img], [0], None, [256], [0, 255])
hist_equa = cv2.calcHist([img_equa], [0], None, [256], [0, 255])
plt.plot(hist, color="b")
plt.plot(hist_equa, color="r")
plt.show()
'''
'''
#! 滤波
# 均值滤波
img_mean = cv2.blur(img, (5,5)) # ksize表示模糊内核大小
cv2.namedWindow('img_mean', cv2.WINDOW_NORMAL)
cv2.imshow('img_mean', img_mean)
# 中值滤波
img_median = cv2.medianBlur(img, 10) # ksize滤波模板的尺寸大小，必须是大于1的奇数
cv2.namedWindow('img_median', cv2.WINDOW_NORMAL)
cv2.imshow('img_median', img_median)
# 高斯滤波
img_Guassian = cv2.GaussianBlur(img,(5,5),0) # ksize高斯内核大小， ksize.width和ksize.height可以不同，但​​它们都必须为正数和奇数，也可以为零，然后根据sigma计算得出
cv2.namedWindow('img_Guassian', cv2.WINDOW_NORMAL)
cv2.imshow('img_Guassian', img_Guassian)
# 双边滤波
img_bilater = cv2.bilateralFilter(img,9,75,75)
cv2.namedWindow('img_bilater', cv2.WINDOW_NORMAL)
cv2.imshow('img_bilater', img_bilater)
'''
'''
#! 二值化处理
#ret,img_threshold = cv2.threshold(img, 127, 255, 0) # 全局二值化处理
img_threshold = myThreshold(img, 127)
cv2.namedWindow('img_threshold', cv2.WINDOW_NORMAL)
cv2.imshow('img_threshold', img_threshold)
'''
'''
#! 边缘检测
# robers算子
Robert_image = myRobert(img)
cv2.namedWindow('Robert_image', cv2.WINDOW_NORMAL)
cv2.imshow('Robert_image', Robert_image)
# sobel 算子
Sobel_image = mySobel(img)
cv2.namedWindow('Sobel_image', cv2.WINDOW_NORMAL)
cv2.imshow('Sobel_image', Sobel_image)
# Laplace算子
Laplace_image = myLaplace(img)
cv2.namedWindow('Laplace_image', cv2.WINDOW_NORMAL)
cv2.imshow('Laplace_image', Laplace_image)
'''
'''
#! 霍夫直线检测
lines = cv2.HoughLinesP(img, 1, np.pi/180, 90, minLineLength=10, maxLineGap=250)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
cv2.namedWindow('img_line', cv2.WINDOW_NORMAL)
cv2.imshow('img_line', img)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()