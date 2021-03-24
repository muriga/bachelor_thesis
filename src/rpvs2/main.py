import fitz
import io
from PIL import Image
from PIL.Image import FLIP_LEFT_RIGHT
import pytesseract
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import stanza
import evaluation
import first
import numpy as np
import cv2
from re import search
import ocr
import pattern

PATH_DATASET = "../../Dataset/"
PATH_LIST = "../../Dataset/all.csv"

def getListOfCompanies(df, pdfNum, text):
    data = {}
    data[df.loc[df['PDF'] == pdfNum]['ICO'].item()] = text
    data_df = pd.DataFrame.from_dict([data])
    data_df.columns = ['text']
    data_df = data_df.sort_index()
    print(data_df)

    cv = CountVectorizer()
    data_cv = cv.fit_transform(data_df.text)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    print(data_dtm)
    #for i, c in enumerate(comedians):
    #    with open("transcripts/" + c + ".txt", "rb") as file:
    #        data[c] = pickle.load(file)


if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    testing = pattern.PatternExtract(PATH_DATASET)
    # testing.itarate_patterns()
    # testing.test()
    #c = first.SimpleClassifier(PATH_DATASET)
    #createTxtFromPdfs('all')
    evaluation.evaluate(testing)