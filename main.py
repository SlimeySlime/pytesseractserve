import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import PIL

import sys
import os

# from easyocr import Reader
import ezocr

CWD = os.getcwd()
IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png']
IMAGE_FILE_PATH = '/img/'
IMAGE_CROPPED_FILE_PATH = '/img/crop/'
FIG_IMG_NUM = 1
cols = 4
rows = 6

def add_subplot_image(figure, src, title, img_num):
    ax1 = figure.add_subplot(rows, cols, img_num) 
    ax1.imshow(src)
    ax1.set_title(title)
    ax1.axis('off')

def do_thing(img_name):

    cwd = os.getcwd()
    img_src = cwd + IMAGE_FILE_PATH + img_name 
    print('do things on ' + img_name)
    plt.style.use('dark_background')
    fig = plt.figure()
    image_to_show = []

    # img = cv2.imread(img_name)
    img = cv2.imread(img_src)
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
        C=9
    )
    image_to_show.append(img_threshed2)

    img_threshed3 = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=9
    )
    image_to_show.append(img_threshed3)

    print(f'current img_threshed.shape : {img_threshed3.shape}')
    # 테두리 검출 
    contours, _ = cv2.findContours(img_threshed3, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
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


    # 테두리 중 번호판 검출 
    # MIN_AREA = 40
    MIN_AREA = 1
    # MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_WIDTH, MIN_HEIGHT = 0.2, 0.8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0
    possible_contours = []
    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
    temp_result = np.zeros((HEIGHT, WIDTH, CHANNEL), dtype=np.uint8)
    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=2)
    image_to_show.append(temp_result)

    def find_chars(contour_list):
        MAX_DIAG_MULTIPLYER = 5
        MAX_ANGLE_DIFF = 12.0
        MAX_AREA_DIFF = 0.5
        MAX_WIDTH_DIFF = 0.8
        MAX_HEIGHT_DIFF = 0.2
        MIN_N_MATCHED = 3

        matched_result_idx = []

        for d1 in contour_list:

            matched_contours_idx = []

            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue
                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx)) 
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                if distance < diagonal_length * MAX_DIAG_MULTIPLYER \
                and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])


            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue
            
            matched_result_idx.append(matched_contours_idx)

            unmatched_contours_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contours_idx.append(d4['idx'])

            unmatched_contours_idx = np.take(possible_contours, unmatched_contours_idx)

            recursive_contour_list = find_chars(unmatched_contours_idx)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break
    
        return matched_result_idx


    result_idx = find_chars(possible_contours)
    print(result_idx)
    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))
    
    temp_result = np.zeros((HEIGHT, WIDTH, CHANNEL), dtype=np.uint8)
    for r in matched_result:
        for d in r:
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=2)
    image_to_show.append(temp_result)

    # 번호판 위치 자르기 
    PLATE_WIDTH_PADDING = 1.5
    PLATE_HEIGHT_PADDING = 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']
        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        # 번호판의 높이와 빗변을 구해서, 기울어진 각도를 계산 
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']]) 
        )
        angle = np.degrees(np.arctan(triangle_height / triangle_hypotenus))
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1)

        # img_rotated = cv2.warpAffine(img_threshed3, M=rotation_matrix, dsize=(WIDTH, HEIGHT))
        img_rotated = cv2.warpAffine(img, M=rotation_matrix, dsize=(WIDTH, HEIGHT))
        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO \
            or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue

        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x' : int(plate_cx - plate_width / 2),
            'y' : int(plate_cy - plate_height / 2),
            'w' : int(plate_width),
            'h' : plate_height
        })
        image_to_show.append(img_cropped)

    # txt = ezocr.read_ocr(img_cropped)
    # print(txt)
    # result_string = pytesseract.image_to_string(img_cropped, lang='Hangul', config='--psm 7 --oem 0')
    result_string = pytesseract.image_to_string(img_cropped, lang='kor', config='--psm 7')
    print('pytesseract img_to_string : ' + result_string)

    cv2.imwrite(cwd + IMAGE_CROPPED_FILE_PATH + img_name, img_cropped)

    for i, img in enumerate(image_to_show):
        add_subplot_image(fig, img, '', i+1)
    plt.show()

    
    

if __name__ == '__main__':
    # do_thing(cwd + IMAGE_FILE_PATH + '34.jpg')
    # do_thing(cwd + IMAGE_FILE_PATH + '65.jpg')

    # do_thing('34.jpg')
    # do_thing('65.jpg')
    do_thing('14.jpg')

    # for img_file in os.listdir(cwd + IMAGE_FILE_PATH):
    #     do_thing(cwd + IMAGE_FILE_PATH + img_file)
    #     break   # for test only one file