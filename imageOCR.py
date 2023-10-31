import math
import cv2
import os
import re
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

class ImageOCR:
    VIDEO_PATH = '/vid/'
    IMAGE_PATH = '/img/ground/'
    FIGURE = plt.figure()
    IMAGE_LENGTH = -1
    FIGURE_ROW = 0
    FIGURE_COL = 0

    IMAGE = None
    HEIGHT = 0
    WIDTH = 0
    CHANNEL = 0
    possible_contours = []

    def gray_and_morph(self, img_src):
        """ 노이즈 제거 및 블러
        설명: 
            노이즈 제거 모폴로지 후
                가우시안 블러
                추가로 모서리 강하게 블러로 탈락
        """
        
        img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

        # 노이즈 제거 모폴로지 
        structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        imgTopHat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, structure)
        imgBlackHat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, structure)

        imgGrayscale_TopHat = cv2.add(img_gray, imgTopHat)
        img_gray_hat = cv2.subtract(imgGrayscale_TopHat, imgBlackHat)

        # 모서리 탈락용 강한 블러 영역
        # x1 = int(self.WIDTH / 3)
        x1 = int(self.WIDTH / 5)
        x2 = self.WIDTH - x1
        y1 = int(self.HEIGHT/ 4)
        y2 = self.HEIGHT - y1
        print(f'x1,2 y1,2 : {x1} {x2} {y1} {y2}')

        # blur_exclude_area = img_gray_hat[y1:y2, x1:x2]
        strong_blurred = cv2.GaussianBlur(img_gray_hat.copy(), ksize=(25, 25), sigmaX=0)
        soft_blurred = cv2.GaussianBlur(img_gray_hat[y1:y2, x1:x2], ksize=(5, 5), sigmaX=0)

        blur_only_edges = strong_blurred.copy()
        blur_only_edges[y1:y2, x1:x2] = soft_blurred
        # img_blurred = cv2.GaussianBlur(img_gray_hat, ksize=(5, 5), sigmaX=0)
        return blur_only_edges

    def threshold(self, img_src, blockSize=11, C=9):
        img_copied = img_src.copy()
        img_threshed = cv2.adaptiveThreshold(
            img_copied,
            maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=blockSize,
            C=C
        )
        return img_threshed

    def get_plate_contours(self, img_src):
        # 테두리 검출 
        contours, _ = cv2.findContours(img_src, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        temp_result = np.zeros((self.HEIGHT, self.WIDTH, self.CHANNEL), dtype=np.uint8)
        cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255,255,255))
        temp_result = np.zeros((self.HEIGHT, self.WIDTH, self.CHANNEL), dtype=np.uint8)
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
        MIN_AREA = 20
        MAX_AREA = 200
        MIN_WIDTH, MIN_HEIGHT = 2, 8
        # MIN_WIDTH, MIN_HEIGHT = 0.2, 0.8
        MIN_RATIO, MAX_RATIO = 0.25, 1.0
        possible_contours = []
        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']
            if area > MIN_AREA and area < MAX_AREA and d['w'] > MIN_WIDTH and \
                d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
                d['idx'] = cnt
                cnt += 1
                possible_contours.append(d)
        # 테두리만 그려서 출력 in temp_result 
        # temp_result = np.zeros((self.HEIGHT, self.WIDTH, self.CHANNEL), dtype=np.uint8)
        # for d in possible_contours:
        #     cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=2)

        # return temp_result
        return possible_contours

    def find_chars(self, contour_list):
        """ plate possible
        Return: 
            possible_idx_list
        """
        MAX_DIAG_MULTIPLYER = 5
        MAX_ANGLE_DIFF = 12.0
        MAX_AREA_DIFF = 0.5
        MAX_WIDTH_DIFF = 0.8
        MAX_HEIGHT_DIFF = 0.5
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

            # unmatched_contours_idx = np.take(possbile_list, unmatched_contours_idx)
            unmatched_contours_idx = np.take(self.possible_contours, unmatched_contours_idx)

            # 재귀적 조사 
            recursive_contour_list = self.find_chars(unmatched_contours_idx)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx

    def cut_plate(self, img_src, matched_list):
        """
        Summary:
            번호판 위치 자르기
        Description:
            cut plate position by contours_list and PLATE RATIO
            PADING TOO
        """
        PLATE_WIDTH_PADDING = 1.5
        PLATE_HEIGHT_PADDING = 1.5
        MIN_PLATE_RATIO = 3
        MAX_PLATE_RATIO = 10

        plate_imgs = []
        plate_infos = []

        for i, matched_chars in enumerate(matched_list):
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
            img_rotated = cv2.warpAffine(img_src, M=rotation_matrix, dsize=(self.WIDTH, self.HEIGHT))
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
                'h' : int(plate_height),
                'cx': int(plate_cx),
                'cy': int(plate_cy),
            })
            # plate_string = pytesseract.image_to_string(img_cropped, lang='kor', config='--psm 7')
            # print(plate_string)
        
        return plate_imgs

    def sharpning(self, img_src, sharp_level):
        """
        설명:
            filter2D 에 
            [-1, -1, -1]
            [-1, sharp_level, -1]
            [-1, -1, -1] 를 적용시켜 리턴
        """
        sharp_array = np.array([[-1, -1, -1],
                                [-1, sharp_level, -1],
                                [-1, -1, -1]])
        sharpend = cv2.filter2D(img_src, -1, sharp_array)
        return sharpend

    def check_plate_regex(self, test_string):
        """
        Description:
            plate regex by r"[0-9]{2}[가-힣][ ][0-9]{3}" 
                -> 64가 3612
        Return: True or False
        """
        regex = r"[0-9]{2}[가-힣][ ][0-9]{3}" 
        result_match = re.match(regex, test_string)    
        if result_match:
            print(test_string + ' is matched')
            return True
        else:
            return False
    # 한글 출력을 위한 PIL 사용
    def draw_kor_text(self, img_src, text_string):
        img_PIL = Image.fromarray(img_src)
        font = ImageFont.truetype('fonts/NanumGothic.ttf', 30)
        draw = ImageDraw.Draw(img_PIL)
        draw.text((160, 170), text_string, font=font, fill=(255,255,255,0))
        return np.array(img_PIL)
    

    def image_list_plot(self, img_list):
        """image_list plot
        설명:
            img_list auto figure plot
        """
        self.IMAGE_LENGTH = len(img_list)
        self.FIGURE_ROW = int(math.sqrt(self.IMAGE_LENGTH))
        col = self.IMAGE_LENGTH / self.FIGURE_ROW 
        if self.IMAGE_LENGTH % self.FIGURE_ROW != 0:
            col += 1
        self.FIGURE_COL = int(col)
        print(f'figure len / row / col = {self.IMAGE_LENGTH} / {self.FIGURE_ROW} / {self.FIGURE_COL}')

        for i, img_info in enumerate(img_list):
            self.IMAGE_LENGTH = len(img_list)
            self.add_subplot_image(img_info[0], img_info[1], i+1)
        plt.show()

    def add_subplot_image(self, src, title, img_num):
        ax1 = self.FIGURE.add_subplot(self.FIGURE_ROW, self.FIGURE_COL, img_num) 
        self.FIGURE.subplots_adjust(left=0.01, bottom=0.01, top=0.99, right=0.99, wspace=0.05, hspace=0.05)
        ax1.imshow(src, cmap='gray')
        ax1.set_title(title)
        ax1.axis('off')

    def img_ocr(self, img_src):
        '''
        설명:
            메인 Image OCR 
        '''
        cwd = os.getcwd()
        img_src_link = cwd + self.IMAGE_PATH + img_src
        print('img_ocr on ' + img_src_link)
        plt.style.use('dark_background')
        self.FIGURE = plt.figure()
        
        images_to_show = []
        img_original = cv2.imread(img_src_link)
        if img_original is None:
            print('no Image !')
            return
        else:
            self.IMAGE = img_original
            self.HEIGHT, self.WIDTH, self.CHANNEL = img_original.shape
            print(f'img_resolution is {self.HEIGHT} x {self.WIDTH} ')
            images_to_show.append([img_original, 'original'])

        img_blurred = self.gray_and_morph(img_original)
        images_to_show.append([img_blurred, 'blurred'])

        img_threshed = self.threshold(img_blurred, blockSize=31, C=9)
        images_to_show.append([img_threshed, 'threshed [b=31, c=9]'])
        # 테두리 index list ( max, min height, width 등으로 )
        self.possible_contours = self.get_plate_contours(img_threshed)
        img_copied = img_original.copy()
        for d in self.possible_contours:
            cv2.rectangle(img_copied,
                        pt1=(d['x'], d['y']), 
                        pt2=(d['x']+d['w'], d['y']+d['h']), 
                        color=(0,255,255), thickness=2 )
        images_to_show.append([img_copied, 'contours'])

        # 다른 block, C 값 임계값으로 테스팅
        img_threshed2 = self.threshold(img_blurred, blockSize=31, C=5)
        images_to_show.append([img_threshed2, 'threshed2 [b=31, c=5]'])
        img_copied = img_original.copy()
        self.possible_contours = self.get_plate_contours(img_threshed2)
        for d in self.possible_contours:
            cv2.rectangle(img_copied,
                        pt1=(d['x'], d['y']), 
                        pt2=(d['x']+d['w'], d['y']+d['h']), 
                        color=(255,0,255), thickness=2 )
        images_to_show.append([img_copied, 'contours2'])

         # 다른 block, C 값 임계값으로 테스팅
        # img_threshed3 = self.threshold(img_blurred, blockSize=31, C=15)
        # images_to_show.append(img_threshed3)
        # self.possible_contours = self.get_plate_contours(img_threshed3)
        # img_copied = img_original.copy()
        # for d in self.possible_contours:
        #     cv2.rectangle(img_copied,
        #                 pt1=(d['x'], d['y']), 
        #                 pt2=(d['x']+d['w'], d['y']+d['h']), 
        #                 color=(255,255,255), thickness=2 )
        # images_to_show.append(img_copied)



        # plate contour 검출 ( 인접 contour와 크기, 각도 등으로 )
        possible_idx = self.find_chars(self.possible_contours)
        print(f'possible_idx.length : {len(possible_idx)}')
        matched_list = []
        for idx_list in possible_idx:
            matched_list.append(np.take(self.possible_contours, idx_list))

        img_copied = img_original.copy()
        for r in matched_list:
            for d in r:
                cv2.rectangle(img_copied,
                        pt1=(d['x'], d['y']), 
                        pt2=(d['x']+d['w'], d['y']+d['h']), 
                        color=(0,255,255), thickness=2 )
        images_to_show.append([img_copied, 'plate chars'])

        print(f'이미지 크기 잠깐 점검 img_src : {self.IMAGE.shape}')
        print(f'이미지 크기 잠깐 점검 img_contours : {img_copied.shape}')

        # plate 검출 및 회전
        possible_plates = self.cut_plate(img_original, matched_list)
        print(f'possible_plates.length {len(possible_plates)}')
        for i, plate_image in enumerate(possible_plates):
            plate_string = pytesseract.image_to_string(plate_image, lang='kor', config='--psm 7')
            images_to_show.append([plate_image, f'plate_cut{i}'])
            print(f'plate_cut {i} ' + plate_string)
            plate_string_conf = pytesseract.image_to_string(plate_image, lang='kor', config='--oem 1 --psm 6')
            print(f'plate_cut {i} - config ' + plate_string_conf)

            sharpend = self.sharpning(plate_image, 9)
            plate_string = pytesseract.image_to_string(sharpend, lang='kor', config='--psm 7')
            images_to_show.append([sharpend, f'plate_sharp{i}'])
            print(f'plate_sharp {i} ' + plate_string)
            plate_string_conf = pytesseract.image_to_string(plate_image, lang='kor', config='--oem 1 --psm 6')
            print(f'plate_sharp {i} - config ' + plate_string_conf)



        # is_plate = self.check_plate_regex(plate_string)

        self.image_list_plot(images_to_show)

        pass 

    def video_ocr(self, vid):
        cwd = os.getcwd()
        video_link = cwd + self.VIDEO_PATH + vid
        print('video ocr on ' + video_link)
        cap = cv2.VideoCapture(video_link)
        if cap is None or not cap.isOpened: 
            print('video read error')
            exit()
        else:
            print(f'start read {vid}')

        ret, frame = cap.read()
        self.HEIGHT, self.WIDTH, self.CHANNEL = frame.shape
        current_frame = 0 
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            current_frame += 1
            # 프레임 스킵
            if current_frame % 5 != 0 or current_frame <= 620: 
                continue

            cv2.putText(frame, f'Frame : {current_frame}', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            img_blurred = self.gray_and_morph(frame)
            img_threshed = self.threshold(img_blurred, blockSize=11, C=9)

            self.possible_contours = self.get_plate_contours(img_threshed)

            # possible_contours = things_in_one(frame) # return possible contours

            # 1. 후보 테두리 그려주기  
            # temp_frame = frame.copy()
            # for d in self.possible_contours:
            #         cv2.rectangle(img_threshed, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(0,255,255), thickness=2)
            

            # 2. plate possible contour index 
            possible_idx = self.find_chars(self.possible_contours)
            matched_result = []
            for idx_list in possible_idx:
                matched_result.append(np.take(self.possible_contours, idx_list))
            
            frame_copy = frame.copy()
            for r in matched_result:
                for d in r:
                    cv2.rectangle(frame_copy, 
                                  pt1=(d['x'], d['y']), 
                                  pt2=(d['x']+d['w'], d['y']+d['h']), 
                                  color=(0,0,255), thickness=2 )
            
            # 3. plate possbile에 대해 crop 후 pytesseract
            frame_copy = frame.copy()
            plate_infos = self.cut_plate(frame, matched_result)
            to_draw_string = ''
            for info in plate_infos:
                cv2.rectangle(frame_copy, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(0,0,255), thickness=2)
            
                frame_plate_cropped = cv2.getRectSubPix(
                    frame_copy,
                    patchSize=(info['w'], info['h']),
                    center=(info['cx'], info['cy'])
                )
                plate_string = pytesseract.image_to_string(frame_plate_cropped, lang='kor', config='--psm 7')
                is_plate = self.check_plate_regex(r"[0-9]{2}[가-힣][ ][0-9]{3}", plate_string)
                if plate_string != '' and is_plate:
                    to_draw_string += plate_string
            
            drawed_frame = self.draw_kor_text(frame_copy, 'plate : ' + to_draw_string)


            cv2.putText(drawed_frame, f'possible_contours : {len(self.possible_contours)}',
                         (20, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('frame', drawed_frame)
            
            # cv2.imshow('frame', temp_frame)   # 
            # cv2.imshow('frame', img_threshed)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



if __name__ == '__main__':
    # video =
    # ocr = VideoOCR()
    # ocr.video_ocr('videoplayback.mp4')
    ocr = ImageOCR()

    # ocr.img_ocr('20230801-110709.jpg')
    # ocr.img_ocr('20230801-111513.jpg')
    # ocr.img_ocr('20230801-out-081324.jpg')
    ocr.img_ocr('/new/1.jpg')
    # ocr.img_ocr('/new/1.jpg')
    ocr.img_ocr('/new/3.jpg')

    pass