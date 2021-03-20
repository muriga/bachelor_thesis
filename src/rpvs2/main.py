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

def getImages(pdf):
    #https://www.thepythoncode.com/article/extract-pdf-images-in-python
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
    #https://www.thepythoncode.com/article/extract-pdf-images-in-python
    pages = len(pdfFile)
    images = []
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
            #image.save(open(f"image{1}_{image_index}.{image_ext}", "wb"))
    return images

def getText(filename):
    pdfFile = fitz.open(filename)
    images = getImages(pdfFile)
    #TODO Moze sa stat, ze najdeme nejaky obrazok - napr podpis - no dokument je v skutocnosti searchable
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

def rotate(image, center = None, scale = 1.0):
    """https://stackoverflow.com/questions/55119504/is-it-possible-to-check-orientation-of-an-image-before-passing-it-through-pytess"""
    angle=360-int(search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def rotation_check(images):
    deskewed = []
    for i in range(len(images)):
        cv_im = np.asarray(images[i])
        newdata = pytesseract.image_to_osd(cv_im)
        rotation = search('(?<=Rotate: )\d+', newdata).group(0)
        print(rotation)
        rotated = images[i]#.transpose(FLIP_LEFT_RIGHT).rotate(-int(rotation))
        #deskewed.append(rotated)
        rotated.save(open(f"imageERROR_{i}.png", "wb"))
    print(extractTextFromImages(deskewed))
    #TODO co ak bol problem v tom, ze ten jeden je zrkadlovy a zakladny deskew vie aj tesseract?

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    ocr.iterate_folder_convert_to_text(PATH_DATASET + "empty",True,True)


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