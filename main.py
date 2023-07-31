import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import PIL

import sys
import os

CWD = os.getcwd()
IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png']
IMAGE_FILE_PATH = '/img/'
FIG_IMG_NUM = 1
cols = 3
rows = 3

def add_subplot_image(figure, src, title, img_num):
    ax1 = figure.add_subplot(rows, cols, img_num) 
    ax1.imshow(src)
    ax1.set_title(title)
    ax1.axis('off')

def do_thing(image_link):
    print('do things on ' + image_link)
    plt.style.use('dark_background')
    fig = plt.figure()
    image_to_show = []

    img = cv2.imread(image_link)
    if img is None:
        print('img is None')
        return
    HEIGHT, WIDTH, CHANNEL = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_to_show.append(img)
    image_to_show.append(img_gray)

    # 노이즈 제거 모폴로지 
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, structure)
    imgBlackHat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, structure)

    imgGrayscale_TopHat = cv2.add(img_gray, imgTopHat)
    img_gray_hat = cv2.subtract(imgGrayscale_TopHat, imgBlackHat)
    image_to_show.append(img_gray_hat)

    # 이미지 블러 처리
    # img_blurred = cv2.GaussianBlur(img_gray, ksize=(5,5), sigmaX=0)
    img_blurred = cv2.GaussianBlur(img_gray_hat, ksize=(5,5), sigmaX=0)
    image_to_show.append(img_blurred)
    
    # 외곽선 검출 Threshold
    img_threshed = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    image_to_show.append(img_threshed)

    img_threshed2 = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=5
    )
    image_to_show.append(img_threshed2)

    img_threshed3 = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=11
    )
    image_to_show.append(img_threshed3)

    contours, _ = cv2.findContours(img_threshed, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    temp_result = np.zeros((HEIGHT, WIDTH, CHANNEL), dtype=np.uint8)
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255,255,255))
    temp_result = np.zeros((HEIGHT, WIDTH, CHANNEL), dtype=np.uint8)
    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2),
        })
    image_to_show.append(temp_result)



    for i, img in enumerate(image_to_show):
        add_subplot_image(fig, img, '', i+1)
    plt.show()
    

if __name__ == '__main__':
    # do_thing("/img/2.jpg")
    cwd = os.getcwd()
    print(cwd)
    print(sys.argv[0])
    for img_file in os.listdir(cwd + IMAGE_FILE_PATH):
        do_thing(cwd + IMAGE_FILE_PATH + img_file)
        # break   # for test only one file