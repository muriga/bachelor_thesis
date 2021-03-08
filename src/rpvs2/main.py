import fitz
import io
from PIL import Image
import pytesseract
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import stanza
import evaluation
import first
import numpy as np

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

def getImages(pdf):
    pages = len(pdf)
    images = []
    for i in range(pages):
        page = pdf[i]
        imageList = page.getImageList()
        for image_index, img in enumerate(page.getImageList(), start=1):
            # get the XREF of the image
            xref = img[0]
            # extract the image bytes
            base_image = pdf.extractImage(xref)
            image_bytes = base_image["image"]
            # get the image extension
            image_ext = base_image["ext"]
            # load it to PIL
            image = Image.open(io.BytesIO(image_bytes))
            # save it to local disk
            images.append(image)
            #image.save(open(f"image{i + 1}_{image_index}.{image_ext}", "wb"))
    return images


def getPdf(path, num):
    filePath = path + num + ".pdf"
    pdfFile = fitz.open(filePath)
    return getImages(pdfFile)


def extractTextFromImages(images):
    txts = ""
    for i in range(len(images)):
        txt = pytesseract.image_to_string(images[i], lang="slk")
        txts += txt
    return txts

def extractTextFromSeearchable(pdfFile):
    txt = ""
    for page in pdfFile.pages():
        txt += page.get_textpage().extractText()
    return txt

def getImages(pdfFile):
    pages = len(pdfFile)
    images = []
    # tato cast je z nejakho tutorialu trochu
    for i in range(pages):
        page = pdfFile[i]
        imageList = page.getImageList()
        for image_index, img in enumerate(page.getImageList(), start=1):
            xref = img[0]
            base_image = pdfFile.extractImage(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    return images

def getText(filename):
    pdfFile = fitz.open(filename)
    images = getImages(pdfFile)
    if len(images) > 0:
        return extractTextFromImages(images)
    return extractTextFromSeearchable(pdfFile)

def isInFolder(folder,file,format=""):
    file = file + format
    for f in os.listdir(folder):
        if os.fsdecode(f) == file:
            return True
    return False


def createTxtFromPdfs(directory_name):
    directory = os.fsencode(PATH_DATASET + directory_name)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        path = PATH_DATASET + directory_name
        if(not isInFolder(path, filename.removesuffix(".pdf"),format=".txt") and filename.endswith(".pdf")):
            text = getText(path + '\\' + filename)
            txtFile = path + '\\' + filename.removesuffix(".pdf") + ".txt"
            with open(txtFile, "w", encoding="utf-8") as file:
                file.write(text)

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
    c = first.SimpleClassifier(PATH_DATASET)
    print(c.is_owner(PATH_DATASET+"majitel/3718"))
    #evaluation.evaluate(first.SimpleClassifier())
    #createTxtFromPdfs('majitel')
    #print("Mame majitela")
    #createTxtFromPdfs('statutar')
    #createTxtFromPdfs('test_majitel')
    #createTxtFromPdfs('test_statutar')