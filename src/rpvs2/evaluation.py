import template
import os
import pandas as pd
from utilities import get_meta_by_pdfname

"""
Tento modul sa stara o testovanie jednotlivých modelov.
"""

PATH_DATASET = "../../Dataset/"
PATH_LIST = "../../Dataset/all.csv"


def load_test_data(majitel, statuar):
    directory = os.fsencode(majitel)
    data = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if (filename.endswith(".pdf")):
            next = ["majitel", filename.removesuffix(".pdf"), ""]
            data.append(next)
    directory = os.fsdecode(statuar)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if (filename.endswith(".pdf")):
            next = ["statutar", filename.removesuffix(".pdf"), ""]
            data.append(next)
    df = pd.DataFrame(data, columns=['typ', 'pdf_name', 'classified_as'])
    return df


def evaluate(c: template.Classifier):
    c.train(PATH_DATASET + "majitel", PATH_DATASET + "statutar")
    #c.train(PATH_DATASET + "majitel", PATH_DATASET + "statutar", path_pretrained="../../models/model_04-30-190526.joblib")
    test_data = load_test_data(PATH_DATASET + "test_majitel", PATH_DATASET + "test_statutar")



    correct_majitel = 0
    correct_statutar = 0
    all_majitel = 0
    all_statutar = 0
    # print(test_data)
    for index, row in test_data.iterrows():
        if c.is_owner(get_meta_by_pdfname(row['pdf_name']), row['pdf_name']):
        # if c.is_owner(row['pdf_name'], row['typ'] == 'majitel'):
            answer = "majitel"
        else:
            answer = "statutar"
        if (row['typ'] == 'majitel'):
            all_majitel += 1
            if answer == 'majitel':
                correct_majitel += 1
            else:
                print(f'{row["pdf_name"]} was not recognized as majitel')
        if (row['typ'] == 'statutar'):
            all_statutar += 1
            if answer == 'statutar':
                # print(row['pdf_name'])
                correct_statutar += 1
            else:
                print(f'{row["pdf_name"]} was not recognized as statutar')
        test_data.at[index, 'classified_as'] = answer
    positive = all_majitel - correct_majitel + correct_statutar
    precision = correct_statutar / positive
    recall = correct_statutar / all_statutar
    f1 = (2 * precision * recall) / (precision + recall)
    majitel = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    result = {"majitel": majitel}
    #print(
    #    f'We had {all_majitel} documents of type "majitel", {correct_majitel / all_majitel * 100}"% of them was classified correctly')
    #print(
    #    f'We had {all_statutar} documents of type "statutar", {correct_statutar / all_statutar * 100}"% of them was classified correctly')
    print(f'On question who is not KUV, just "statutar". Precision = {precision}, recall = {recall}, f1-score = {f1}')
    precision2 = correct_majitel / (all_statutar - correct_statutar + correct_majitel)
    recall2 = correct_majitel / all_majitel
    f12 = (2 * precision2 * recall2) / (precision2 + recall2)
    statutar = {
        "precision": precision2,
        "recall": recall2,
        "f1": f12
    }
    result["statutar"] = statutar
    print(
        f'On question who is KUV, "majitel". Precision = {precision2}, recall = {recall2}, f1-score = {f12}')
    c.write_desc(result)
