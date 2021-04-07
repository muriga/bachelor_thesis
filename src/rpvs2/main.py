import pytesseract
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import slearning
import ocr
import pattern
import evaluation

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
    # ocr.iterate_folder_convert_to_text(PATH_DATASET + 'test_statutar',save=True,contains_txt=False)
    #ocr.iterate_folder_convert_to_text(PATH_DATASET + "majitel", save=True, contains_txt=True)
    #ocr.iterate_folder_convert_to_text(PATH_DATASET + "statutar", save=True, contains_txt=True)
    #ocr.iterate_folder_convert_to_text(PATH_DATASET + "test_majitel", save=True, contains_txt=True)
    #ocr.iterate_folder_convert_to_text(PATH_DATASET + "test_statutar", save=True, contains_txt=False)
    #testing.pattern_statistics()
    #slearning.find_model(PATH_DATASET)
    a = slearning.SupervisedClassifier(PATH_DATASET)
    evaluation.evaluate(a)
    # testing = pattern.PatternExtract(PATH_DATASET)
    # evaluation.evaluate(testing)