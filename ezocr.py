from easyocr import Reader
import os
import numpy as np

import testground
import pytesseract

IMAGE_FILE_PATH = '/img/'

def read_ocr(img_link):
    # print('ocr on ' + img_link )
    reader = Reader(lang_list=['ko'])
    result = reader.readtext(img_link)
    print(result)
    return result

def read_tesseract(img_link):
    print('tesseract on ' + img_link)
    result = pytesseract.image_to_string(img_link, lang='kor')
    print('result is ' + result)

if __name__ ==  '__main__':
    cwd = os.getcwd()
    print(cwd)
    # read_ocr(cwd + IMAGE_FILE_PATH + '34.jpg')
    read_tesseract(cwd + IMAGE_FILE_PATH + '34.jpg')
    read_tesseract(cwd + IMAGE_FILE_PATH + 'panel.png')

