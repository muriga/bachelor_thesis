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


def clean_stopwords():
    #https://code.google.com/archive/p/stop-words/downloads
    df = pd.read_fwf(PATH_DATASET + "stop-words-slovak.txt", encoding="utf-8")
    df2 = pd.read_fwf(PATH_DATASET + "stop-words-slovak.txt", encoding="utf-8")
    df = pd.concat([df, df2]).drop_duplicates()
    df.to_csv(r'faza1.csv')
    #print(df)
    nlp = stanza.Pipeline('sk', verbose=False, processors="tokenize,lemma")
    annotated_stop_words = nlp(df.to_string(header=False,index=False))
    lemmatized_stopwords = []
    for word in annotated_stop_words.sentences[0].words:
        lemmatized_stopwords.append(word.lemma)
    df_lemmatized_stopwords = pd.DataFrame(lemmatized_stopwords, columns=["word"]).drop_duplicates()
    df_lemmatized_stopwords.to_csv(f'{PATH_DATASET}stop-words.txt',header=False,index=False)

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    #ocr.iterate_folder_convert_to_text(PATH_DATASET + "empty",True,True)
    #d = dict()
    #d = ocr.iterate_folder_get_text(PATH_DATASET + "empty")
    #print(d.keys())
    #print(d)
    print(ocr.get_text(PATH_DATASET + "empty/37158.txt"))

    #print(getText(PATH_DATASET + "statutar/119328.pdf"))
    #rotation_check(getImages(fitz.open(PATH_DATASET + "statutar/119773.pdf")))
    #rotation_check(getImages(fitz.open(PATH_DATASET + "statutar/119662.pdf")))
    #c = first.SimpleClassifier(PATH_DATASET)
    #print(c.is_owner(PATH_DATASET+"majitel/3718"))
    #createTxtFromPdfs('all')
    #evaluation.evaluate(c)
    #createTxtFromPdfs('majitel')
    #print("Mame majitela")
    #createTxtFromPdfs('statutar')
    #createTxtFromPdfs('test_majitel')
    #createTxtFromPdfs('test_statutar')