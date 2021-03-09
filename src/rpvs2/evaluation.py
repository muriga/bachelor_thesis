import template
import os
import pandas as pd
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
            next = ["statutar", filename, ""]
            data.append(next)
    df = pd.DataFrame(data,columns=['typ','pdf_name','classified_as'])
    return df


def evaluate(c: template.Classifier):
    c.train(PATH_DATASET + "majitel", PATH_DATASET + "statutar")
    test_data = load_test_data(PATH_DATASET + "majitel", PATH_DATASET + "statutar")
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    stats = {}
    correct_majitel = 0
    correct_statutar = 0
    all_majitel = 0
    all_statutar = 0
    #print(test_data)
    for index, row in test_data.iterrows():
        if c.is_owner(row['pdf_name']):
            answer = "majitel"
        else:
            answer = "statutar"
        if(row['typ'] == 'majitel'):
            all_majitel += 1
            if answer == 'majitel':
                correct_majitel += 1
        if(row['typ'] == 'statutar'):
            all_statutar += 1
            if answer == 'statutar':
                correct_statutar += 1
        test_data.at[index,'classified_as'] = answer
    print(f'We had {all_majitel} documents of type "majitel", {correct_majitel/all_majitel*100}"% of them was classified correctly')
    print(f'We had {all_statutar} documents of type "statutar", {correct_statutar/all_statutar*100}"% of them was classified correctly')