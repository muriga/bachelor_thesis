import os
from slearning import SupervisedClassifier
import csv
import pandas as pd
from webscraping import process_detail_page

PATH_DATASET = "../../Dataset/"
RESULTS_FOLDER_PATH = "vysledky"
CHECKED_CSV_PATH = RESULTS_FOLDER_PATH + "/skontrolovane.csv"
PATH_MODEL = '../../models/model_04-17-082028.joblib'
NEREGISTROVANY = 0
MAJITEL = 1
STATUTAR = 2


class Handler:

    def __init__(self):
        self.classifier = SupervisedClassifier(PATH_DATASET)
        self.classifier.train(PATH_DATASET + "majitel", PATH_DATASET + "statutar", path_pretrained=PATH_MODEL)

    def start_findig_statutar(self, to_num_pvs: int):
        if not os.path.exists(RESULTS_FOLDER_PATH):
            os.makedirs(RESULTS_FOLDER_PATH)
        self.continue_finding_statutar(1, to_num_pvs)

    def continue_where_stopped(self, amount_to_process):
        if not os.path.exists(RESULTS_FOLDER_PATH):
            os.makedirs(RESULTS_FOLDER_PATH)
        last_row = pd.read_csv(CHECKED_CSV_PATH).tail(1)
        last_read = last_row.iloc[0][6]
        self.continue_finding_statutar(last_read, to_num_pvs=last_read + amount_to_process)

    def continue_finding_statutar(self, from_num_pvs: int, to_num_pvs: int):
        for i in range(from_num_pvs, to_num_pvs):
            meta_data, pdf = process_detail_page(i)
            if pdf is None:
                result = NEREGISTROVANY
            elif self.classifier.is_owner(meta_data, pdf):
                result = MAJITEL
            else:
                result = STATUTAR
            save(meta_data, pdf, result)


def save(meta_data, pdf, result):
    if result == MAJITEL:
        state = "majitel"
    elif result == STATUTAR:
        state = "statutar"
    else:
        state = "neregistrovany"
    meta_data.append(state)
    with open(CHECKED_CSV_PATH, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, dialect='excel')
        writer.writerow(meta_data)
    if result == STATUTAR:
        pdf_name = RESULTS_FOLDER_PATH + '/' + str(meta_data[9]) + ".pdf"
        pdf.save(pdf_name)

