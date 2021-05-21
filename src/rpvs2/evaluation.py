import template
import os
import pandas as pd
from ocr import iterate_folder_get_text
from utilities import get_meta_by_pdfname
from utilities import replace_meta
from sklearn.model_selection import KFold
from slearning import MLPClassifierBoW

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


def k_fold():
    documents_majitelia = iterate_folder_get_text(PATH_DATASET + "majitel") | iterate_folder_get_text(PATH_DATASET + "test_majitel")
    documents_statutari = iterate_folder_get_text(PATH_DATASET + "statutar") | iterate_folder_get_text(PATH_DATASET + "test_statutar")
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
    # documents = documents_statutari | documents_majitelia
    targets = [0] * len(documents_majitelia) + [1] * len(documents_statutari)
    # for key in documents:
    #     data = get_meta_by_pdfname(key)
    #     data['k_fold'] = True
    #     documents[key] = replace_meta(documents[key], data)
    #     meta_data.append(data)
    # documents = list(documents.values())
    kf = KFold(n_splits=5, random_state=5, shuffle=True)

    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for training_indices, testing_indices in kf.split(documents, targets):
        k_documents = []
        k_targets = []
        answers = {
            "majitel": {"majitel" : 0,
                        "statutar" : 0},
            "statutar": {"statutar" : 0,
                        "majitel" : 0}
        }
        for i in training_indices:
            k_documents.append(documents[i])
            k_targets.append(targets[i])
        classifier = MLPClassifierBoW(PATH_DATASET)
        classifier.train(None,None,using_k_fold=True, loaded_texts=k_documents, loaded_targets=k_targets)
        for i in testing_indices:
            ans = classifier.is_owner(meta_data[i], documents[i])
            # if not ans and targets[i] == 0:
            #     print(f'{keys[i]} nebol klasifinovany ako majitel')
            # elif ans and targets[i] == 1:
            #     print(f'{keys[i]} nebol klasifinovany ako statutar')
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
        precision = answers["statutar"]["statutar"] / (answers["statutar"]["statutar"] + answers["statutar"]["majitel"])
        recall = answers["statutar"]["statutar"] / (answers["statutar"]["statutar"] + answers["majitel"]["statutar"])
        f1 = (2 * precision * recall) / (precision + recall)
        print(f'precision: {precision}\nrecall: {recall}\nf1: {f1}\n\n')
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    total_f1 /= 5
    total_recall /= 5
    total_precision /= 5
    print(f'precision: {total_precision}\nrecall: {total_recall}\nf1: {total_f1}\n\n')
    #data = list(documents.values())


    #return data, target, [*documents]
