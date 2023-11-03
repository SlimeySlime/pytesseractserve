import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
# import easyocr
from paddleocr import PaddleOCR, draw_ocr
from .regex_util import plate_regex

pOCR = PaddleOCR(lang='korean')

def ocr(frame):
    result = pOCR.ocr(frame, cls=True)
    # result2 = paddle.ocr(frame, cls=True)
    return result


def video_stream(vid_link):
    vid_path = os.getcwd() + vid_link
    print(f'read video from ${os.getcwd() + vid_link}')
    if os.path.isfile(vid_path):
        cap = cv2.VideoCapture(vid_path)
    else:
        print('파일이 없습니다')
        return
        
    if cap is None or not cap.isOpened: 
        print('video read error')
        exit()
    else:
        print(f'start read {cap}')
    
    current_frame = 0 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 스킵
        current_frame += 1
        skip_frame_late = 5
        if current_frame % skip_frame_late != 0: 
            continue

        HEIGHT, WIDTH, CHANNEL = frame.shape


        result = ocr(frame)
        print(result)

        guess = None
        if result is not None:
            for item in result:
                guess = item[1]
                # print(item[1])
                
        cv2.putText(frame, f'result : {guess}', (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Frame : {current_frame}', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    test_vid = '/img/new/91다4090.mp4'
    # test_vid = '/img/new/91부5087.mp4'
    video_stream(test_vid)

    pass