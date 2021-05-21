import pytesseract
import ui
import evaluation
import slearning

PATH_DATASET = "../../Dataset/"
PATH_LIST = "../../Dataset/all.csv"


if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    # ocr.iterate_folder_convert_to_text(PATH_DATASET + 'test_statutar',save=True,contains_txt=False)
    # ocr.iterate_folder_convert_to_text(PATH_DATASET + "majitel", save=True, contains_txt=True)
    # ocr.iterate_folder_convert_to_text(PATH_DATASET + "statutar", save=True, contains_txt=True)
    # ocr.iterate_folder_convert_to_text(PATH_DATASET + "test_majitel", save=True, contains_txt=True)
    # ocr.iterate_folder_convert_to_text(PATH_DATASET + "all", save=True, contains_txt=True)

    # t = slearning.SentenceClassifier(PATH_DATASET + "sentences.csv")
    # t.train()


    testing = slearning.MLPClassifierBoW(PATH_DATASET)
    # evaluation.evaluate(testing)
    # testing_pattern = pattern.PatternExtract(PATH_DATASET)
    # evaluation.evaluate(testing_pattern)

    # t = slearning.MLPClassifierWSent(PATH_DATASET)
    # evaluation.evaluate(t)


    evaluation.k_fold()
    # a = ui.Handler()
    # while a.handle():
    #     pass
    #a.continue_where_stopped(2)