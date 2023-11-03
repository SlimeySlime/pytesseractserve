import easyocr
import matplotlib.pyplot as plt
import matplotlib.image as mping
import cv2
import os
import math

# result =
class EzOCR():

    IMAGE = None
    OCR_RESULT = None

    reader = easyocr.Reader(['ko'], gpu=True)
    figure = plt.figure()

    def read_image(self, img_src):
        cwd = os.getcwd()
        img_link = cwd + img_src
        img_original = cv2.imread(img_link)
        self.IMAGE = img_original

    def draw_rect(self, result):
        if self.IMAGE is not None:
            img_copied = self.IMAGE.copy()
        else: 
            return
        
        for i in result:
            x = i[0][0][0]
            y = i[0][0][1]
            w = i[0][1][0] - x
            h = i[0][2][1] - i[0][1][1]
            cv2.rectangle(img_copied,
                pt1=(x,y), pt2=(x+w, y+h), color=(255,0,255), thickness=2
            )
            print(f'draw rect xywh: ${x} ${y} ${w} ${h}')
            


        return img_copied

    def ocr(self, img_link):

        self.read_image(img_link)
        self.OCR_RESULT = self.reader.readtext(self.IMAGE)
        for res in self.OCR_RESULT:
            print(f'result {res[1]} like {round(res[2], 3)} ')
            # print(res)


        return

if __name__ == '__main__':
    ez = EzOCR()
    ez.ocr('/img/new/0299.png') # 97버0299
    ez.ocr('/img/new/5103.png') # 96거5103
    ez.ocr('/img/new/7302.png') # 89어7302
    ez.ocr('/img/new/8712.png') # 83두8712

# plt.show()



