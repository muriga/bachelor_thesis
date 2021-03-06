import os
import pandas as pd
from ocr import iterate_folder_get_text
from utilities import get_meta_by_pdfname
from utilities import replace_meta
from sklearn.model_selection import KFold
from slearning import MLPClassifierBoW
from utilities import Classifier
import csv

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


def evaluate(c: Classifier):
    c.train(PATH_DATASET + "majitel", PATH_DATASET + "statutar")

    test_data = load_test_data(PATH_DATASET + "test_majitel", PATH_DATASET + "test_statutar")

    correct_majitel = 0
    correct_statutar = 0
    all_majitel = 0
    all_statutar = 0
    for index, row in test_data.iterrows():
        if c.is_owner(get_meta_by_pdfname(row['pdf_name']), row['pdf_name']):
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


def k_fold(hidden=(75, 50), d=None, k=None, m=None, t=None, df=(0.25, 0.85)):
    print(hidden)
    if d is None:
        documents_majitelia = iterate_folder_get_text(PATH_DATASET + "majitel")
        documents_statutari = iterate_folder_get_text(PATH_DATASET + "statutar")
        documents = []
        keys = []
        meta_data = []
        for k, v in documents_majitelia.items():
            data = get_meta_by_pdfname(k)
            data['k_fold'] = True
            v = replace_meta(v, data)
            documents.append(v)
            keys.append(k)
            meta_data.append(data)
        for k, v in documents_statutari.items():
            data = get_meta_by_pdfname(k)
            data['k_fold'] = True
            v = replace_meta(v, data)
            documents.append(v)
            keys.append(k)
            meta_data.append(data)
        targets = [0] * len(documents_majitelia) + [1] * len(documents_statutari)
    else:
        documents = d
        meta_data = m
        targets = t
    kf = KFold(n_splits=5, random_state=5, shuffle=True)

    total_precision, majitel_total_precision = 0, 0
    total_recall, majitel_total_recall = 0, 0
    total_f1, majitel_total_f1 = 0, 0

    for training_indices, testing_indices in kf.split(documents, targets):
        k_documents = []
        k_targets = []
        answers = {
            "majitel": {"majitel": 0,
                        "statutar": 0},
            "statutar": {"statutar": 0,
                         "majitel": 0}
        }
        for i in training_indices:
            k_documents.append(documents[i])
            k_targets.append(targets[i])
        classifier = MLPClassifierBoW(PATH_DATASET, v_min_df=df[0], v_max_df=df[1])
        classifier.train(None, None, using_k_fold=True, loaded_texts=k_documents, loaded_targets=k_targets,
                         hidden=hidden)
        for i in testing_indices:
            ans = classifier.is_owner(meta_data[i], documents[i])

            if ans:
                if targets[i] == 0:
                    answers['majitel']['majitel'] += 1
                else:
                    answers['majitel']['statutar'] += 1
            else:
                if targets[i] == 1:
                    answers['statutar']['statutar'] += 1
                else:
                    answers['statutar']['majitel'] += 1
        try:
            statuter_precision = answers["statutar"]["statutar"] / (
                answers["statutar"]["statutar"] + answers["statutar"]["majitel"])
            statutar_recall = answers["statutar"]["statutar"] / (
                    answers["statutar"]["statutar"] + answers["majitel"]["statutar"])
            statutar_f1 = (2 * statuter_precision * statutar_recall) / (statuter_precision + statutar_recall)
        except ZeroDivisionError:
            statutar_f1 = 0
            statutar_recall = 0
            statuter_precision = 0

        try:
            majitel_precision = answers["majitel"]["majitel"] / (
                    answers["majitel"]["majitel"] + answers["majitel"]["statutar"])
            majitel_recall = answers["majitel"]["majitel"] / (
                    answers["majitel"]["majitel"] + answers["statutar"]["majitel"])
            majitel_f1 = (2 * majitel_precision * majitel_recall) / (majitel_precision + majitel_recall)
        except ZeroDivisionError:
            majitel_f1 = 0
            majitel_recall = 0
            majitel_precision = 0

        total_precision += statuter_precision
        total_recall += statutar_recall
        total_f1 += statutar_f1
        majitel_total_precision += majitel_precision
        majitel_total_recall += majitel_recall
        majitel_total_f1 += majitel_f1
    print(f'Statutar\nprecision: {total_precision / 5}\nrecall: {total_recall / 5}\nf1: {total_f1 / 5}\n\n')

    return total_f1 / 5, total_precision / 5, total_recall / 5


def finding_model():
    documents_majitelia = iterate_folder_get_text(PATH_DATASET + "majitel")
    documents_statutari = iterate_folder_get_text(PATH_DATASET + "statutar")
    documents = []
    keys = []
    meta_data = []
    for k, v in documents_majitelia.items():
        data = get_meta_by_pdfname(k)
        data['k_fold'] = True
        v = replace_meta(v, data)
        documents.append(v)
        keys.append(k)
        meta_data.append(data)
    for k, v in documents_statutari.items():
        data = get_meta_by_pdfname(k)
        data['k_fold'] = True
        v = replace_meta(v, data)
        documents.append(v)
        keys.append(k)
        meta_data.append(data)
    targets = [0] * len(documents_majitelia) + [1] * len(documents_statutari)

    best = [
        (15,),
        (50,),
        (75,),
        (100,),
        (120,),
        (140,),
        (15, 15),
        (50, 15),
        (75, 15),
        (100, 15),
        (120, 15),
        (140, 15),
        (15, 25),
        (50, 25),
        (75, 25),
        (100, 25),
        (120, 25),
        (140, 25),
        (25, 50),
        (50, 50),
        (100, 50),
        (120, 50),
        (140, 50),
        (25, 75),
        (50, 75),
        (75, 75),
        (100, 75),
        (125, 75),
        (25, 100),
        (50, 100),
        (75, 100),
        (100, 100),
        (140, 100),
        (15, 120),
        (25, 120),
        (75, 120),
        (100, 120),
        (120, 120),
        (15, 15, 15),
        (25, 15, 15),
        (25, 15, 15),
        (50, 15, 15),
        (50, 25, 15),
        (75, 25, 15),
        (100, 25, 15),
        (120, 25, 15),
        (15, 50, 15),
        (25, 50, 15),
        (50, 50, 15),
        (75, 50, 15),
        (100, 50, 15),
        (120, 50, 15),
        (15, 75, 15),
        (25, 75, 15),
        (50, 75, 15),
        (100, 75, 15),
        (15, 100, 15),
        (25, 100, 15),
        (50, 100, 15),
        (75, 100, 15),
        (100, 100, 15),
        (15, 120, 15),
        (25, 120, 15),
        (50, 120, 15),
        (75, 120, 15),
        (140, 120, 15),
        (15, 140, 15),
        (25, 140, 15),
        (75, 140, 15),
        (120, 140, 15),
        (140, 140, 15),
        (15, 15, 25),
        (15, 25, 25),
        (25, 25, 25),
        (75, 25, 25),
        (100, 25, 25),
        (120, 25, 25),
        (25, 50, 25),
        (50, 50, 25),
        (75, 50, 25),
        (100, 50, 25),
        (120, 50, 25),
        (140, 50, 25),
        (15, 75, 25),
        (50, 75, 25),
        (75, 75, 25),
        (100, 75, 25),
        (120, 75, 25),
        (140, 75, 25),
        (15, 100, 25),
        (25, 100, 25),
        (50, 100, 25),
        (75, 100, 25),
        (120, 100, 25),
        (140, 100, 25),
        (25, 120, 25),
        (50, 120, 25),
        (25, 140, 25),
        (50, 140, 25),
        (100, 140, 25),
        (15, 50, 50),
        (25, 50, 50),
        (50, 50, 50),
        (120, 50, 50),
        (25, 75, 50),
        (50, 75, 50),
        (15, 75, 50),
        (25, 100, 50),
        (50, 100, 50),
        (75, 100, 50),
        (100, 100, 50),
        (120, 100, 50),
        (25, 120, 50),
        (50, 120, 50),
        (100, 120, 50),
        (15, 140, 50),
        (25, 140, 50),
        (75, 140, 50),
        (120, 140, 50),
        (140, 140, 50),
        (15, 15, 75),
        (25, 15, 75),
        (50, 15, 75),
        (75, 15, 75),
        (100, 15, 75),
        (120, 15, 75),
        (140, 15, 75),
        (15, 25, 75),
        (25, 25, 75),
        (50, 25, 75),
        (75, 25, 75),
        (100, 25, 75),
        (120, 25, 75),
        (25, 50, 75),
        (50, 50, 75),
        (100, 50, 75),
        (140, 50, 75),
        (25, 75, 75),
        (75, 75, 75),
        (120, 75, 75),
        (140, 75, 75),
        (15, 100, 75),
        (25, 100, 75),
        (50, 100, 75),
        (75, 100, 75),
        (120, 100, 75),
        (25, 120, 75),
        (50, 120, 75),
        (120, 120, 75),
        (140, 120, 75),
        (25, 140, 75),
        (50, 140, 75),
        (75, 140, 75),
        (120, 140, 75),
        (140, 140, 75),
        (15, 15, 100),
        (25, 15, 100),
        (50, 15, 100),
        (120, 15, 100),
        (15, 25, 100),
        (75, 25, 100),
        (100, 25, 100),
        (120, 25, 100),
        (140, 25, 100),
        (25, 50, 100),
        (50, 50, 100),
        (140, 50, 100),
        (15, 75, 100),
        (100, 75, 100),
        (120, 75, 100),
        (140, 75, 100),
        (25, 100, 100),
        (140, 100, 100),
        (15, 120, 100),
        (25, 120, 100),
        (75, 120, 100),
        (100, 120, 100),
        (120, 120, 100),
        (140, 120, 100),
        (25, 140, 100),
        (75, 140, 100),
        (100, 140, 100),
        (140, 140, 100)
    ]

    tak = [(15, 15)]
    df = [
        (0.04, 0.98)
    ]
    num=0
    results = []
    r_a, r_b, r_c = [], [], []
    # for df_i in df:
    #     for i in best:
    #         a, b, c = k_fold(i, k=keys, d=documents, t=targets, m=meta_data, df=df_i)
    #         r_a.append(a)
    #         r_b.append(b)
    #         r_c.append(c)
    #     with open(f'eval2_{num}.csv', 'w', newline='', encoding='utf-8') as file:
    #         writer = csv.writer(file, dialect='excel')
    #         for j in zip(best, r_a, r_b, r_c):
    #             writer.writerow(j)
    #     num += 1

    for i in best:
        results.append(k_fold(i, d=documents, t=targets, m=meta_data))
    with open("eval2.csv", 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, dialect='excel')
        for i in zip(best, results):
            writer.writerow((i))
