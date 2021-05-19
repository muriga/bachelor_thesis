import pytesseract
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import slearning
import ocr
import pattern
import evaluation
import os
import ui

PATH_DATASET = "../../Dataset/"
PATH_LIST = "../../Dataset/all.csv"


def get_save_relation_sentences(p: pattern.PatternExtract):
    path_dataset = "../../Dataset/"
    files = dict()
    directory = os.fsencode(path_dataset + "majitel")
    majitelia = []
    statutari = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            majitelia.append(filename.title().removesuffix(".Txt"))
    directory = os.fsencode(path_dataset + "statutar")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            statutari.append(filename.title().removesuffix(".Txt"))
    list_of_tuples = []
    list_text = []
    list_tokenized_text = []
    list_label = []
    for i in majitelia:
        text, tokenized_text, label = p.play(i, r'majitel')
        list_text.extend(text)
        list_tokenized_text.extend(tokenized_text)
        list_label.extend(label)
    for i in statutari:
        text, tokenized_text, label  = p.play(i, r'statutar')
        list_text.extend(text)
        list_tokenized_text.extend(tokenized_text)
        list_label.extend(label)

    list_of_tuples = list(zip(list_text, list_tokenized_text, list_label))
    df = pd.DataFrame(list_of_tuples, columns=['Sentence', 'Tokenized', 'Class'])
    df.to_csv("sentences.csv")


if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    # ocr.iterate_folder_convert_to_text(PATH_DATASET + 'test_statutar',save=True,contains_txt=False)
    # ocr.iterate_folder_convert_to_text(PATH_DATASET + "majitel", save=True, contains_txt=True)
    # ocr.iterate_folder_convert_to_text(PATH_DATASET + "statutar", save=True, contains_txt=True)
    # ocr.iterate_folder_convert_to_text(PATH_DATASET + "test_majitel", save=True, contains_txt=True)
    # ocr.iterate_folder_convert_to_text(PATH_DATASET + "all", save=True, contains_txt=True)

    # t = slearning.SentenceClassifier(PATH_DATASET + "sentences.csv")
    # t.train()


    # testing = slearning.MLPClassifierBoW(PATH_DATASET)
    # evaluation.evaluate(testing)
    # testing_pattern = pattern.PatternExtract(PATH_DATASET)
    # evaluation.evaluate(testing_pattern)

    t = slearning.MLPClassifierWSent(PATH_DATASET)
    evaluation.evaluate(t)

    #a = ui.Handler()
    #a.continue_where_stopped(2)