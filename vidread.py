import cv2
import os
import numpy as np

VIDEO_PATH = '/vid/'
HEIGHT = 0
WIDTH = 0
CHANNEL = 0

def gray_and_morph(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # 노이즈 제거 모폴로지 
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, structure)
    imgBlackHat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, structure)

    imgGrayscale_TopHat = cv2.add(img_gray, imgTopHat)
    img_gray_hat = cv2.subtract(imgGrayscale_TopHat, imgBlackHat)

    # 블러 
    # img_blurred = cv2.GaussianBlur(img_gray, ksize=(5,5), sigmaX=0)
    img_blurred = cv2.GaussianBlur(img_gray_hat, ksize=(5,5), sigmaX=0)
    # print(img_blurred)
    return img_blurred

def threshold(img_src, blockSize=11, C=9):
    img_threshed = cv2.adaptiveThreshold(
        img_src,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=blockSize,
        C=C
    )
    return img_threshed

def get_plate_contours(img_src):
    # 테두리 검출 
    contours, _ = cv2.findContours(img_src, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    temp_result = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255,255,255))
    temp_result = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    contours_dict = []
    # 모든 테두리 저장 
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
    # 테두리 중 번호판 검출 
    MIN_AREA = 40
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    # MIN_WIDTH, MIN_HEIGHT = 0.2, 0.8
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

    return temp_result

def things_in_one(img_src):
    HEIGHT, WIDTH, CHANNEL = img_src.shape
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # 노이즈 제거 모폴로지 
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, structure)
    imgBlackHat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, structure)

    imgGrayscale_TopHat = cv2.add(img_gray, imgTopHat)
    img_gray_hat = cv2.subtract(imgGrayscale_TopHat, imgBlackHat)

    # 이미지 블러 처리
    # img_blurred = cv2.GaussianBlur(img_gray, ksize=(5,5), sigmaX=0)
    img_blurred = cv2.GaussianBlur(img_gray_hat, ksize=(5,5), sigmaX=0)
    
    # 외곽선 검출 Threshold
    img_threshed = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )

    # 테두리 검출 
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
    
    # 테두리 중 번호판 검출 
    MIN_AREA = 40
    # MIN_AREA = 1
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    # MIN_WIDTH, MIN_HEIGHT = 0.2, 0.8
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
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,0,0), thickness=2)

    # return temp_result
    return possible_contours


def find_chars(contour_list):

    original_contours = contour_list.copy()
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

        # unmatched_contours_idx = np.take(possible_contours, unmatched_contours_idx)
        unmatched_contours_idx = np.take(contour_list, unmatched_contours_idx)

        recursive_contour_list = find_chars(unmatched_contours_idx)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx

def video_ocr(vid):
    cwd = os.getcwd()
    video_link = cwd + VIDEO_PATH + vid
    print('video ocr on ' + video_link)
    cap = cv2.VideoCapture(video_link)
    if cap is None or not cap.isOpened: 
        print('video read error')
        exit()
    else:
        print(f'start read {vid}')
    
    current_frame = 0 
    # ret, frame = cap.read()

    # HEIGHT, WIDTH, CHANNEL = frame.shape
    # print(f'Frame.shaep : {HEIGHT}, {WIDTH}, {CHANNEL}')

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        current_frame += 1
        # 프레임 스킵
        if current_frame % 5 != 0 or current_frame <= 600: 
            continue

        cv2.putText(frame, f'Frame : {current_frame}', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # img_blurred = gray_and_morph(frame)
        # img_threshed = threshold(img_blurred)
        # contours = get_plate_contours(img_threshed)
        possible_contours = things_in_one(frame) # return possible contours

        # temp_frame = frame.copy()
        # for d in possbible_contours:
        #         cv2.rectangle(temp_frame, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(0,0,255), thickness=2)
        
        temp_result = find_chars(possible_contours)
        matched_result = []
        for idx_list in temp_result:
            matched_result.append(np.take(possible_contours, idx_list))
        
        frame_copy = frame.copy()
        for r in matched_result:
            for d in r:
                cv2.rectangle(frame_copy, 
                              pt1=(d['x'], d['y']), 
                              pt2=(d['x']+d['w'], d['y']+d['h']), 
                              color=(0,0,255), thickness=2 )
        
        cv2.imshow('frame', temp_result)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == '__main__':
    # video =
    video_ocr('videoplayback.mp4')

    pass