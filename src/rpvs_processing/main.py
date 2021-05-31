import pytesseract
import ui
import evaluation
import slearning

PATH_DATASET = "../../Dataset/"
PATH_LIST = "../../Dataset/all.csv"

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

    # testing = slearning.MLPClassifierBoW(PATH_DATASET)
    # evaluation.evaluate(testing)
    # testing_pattern = pattern.PatternExtract(PATH_DATASET)
    # evaluation.evaluate(testing_pattern)
    # testing_w_sentences = slearning.MLPClassifierWSent(PATH_DATASET)
    # evaluation.evaluate(testing_w_sentences)

    a = ui.Handler()
    while a.handle():
        pass
