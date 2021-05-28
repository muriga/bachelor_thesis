import pytesseract
import ui
import evaluation
import slearning
import re
import csv

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

    # testing = slearning.MLPClassifierBoW(PATH_DATASET)
    # evaluation.evaluate(testing)
    # testing_pattern = pattern.PatternExtract(PATH_DATASET)
    # evaluation.evaluate(testing_pattern)

    # t = slearning.MLPClassifierWSent(PATH_DATASET)
    # evaluation.evaluate(t)
    evaluation.finding_model()


    # evaluation.k_fold()
    # a = ui.Handler()
    # while a.handle():
    #     pass
    # a.continue_where_stopped(2)


def fix():
    with open(f'parse.txt', 'r', newline='', encoding='utf-8') as file:
        txt = file.read()
    records = re.findall("\(.*\).*\n.*:\t[0-9.]*", txt)
    print(len(records))
    eval = []
    num = 0
    for record in records:
        hidden_layer = re.search("\(.*\)", record)
        f1 = re.search("0\.[0-9]*", record)
        eval.append((hidden_layer.group(0), f1.group(0)))
        if (len(eval) == 161):
            with open(f'fix_{num}.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, dialect='excel')
                for j in eval:
                    writer.writerow(j)
            eval = []
            num += 1