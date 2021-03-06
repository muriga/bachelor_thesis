"""Module for different utilities for this project"""

import pandas as pd
import stanza
import re
import fitz

PATH_DATASET = "../../Dataset/"
PATH_LIST = "../../Dataset/all.csv"


def clean_stopwords():
    """Little script which took stop-words, lemmatized them using stanza and drop duplicates"""
    # https://code.google.com/archive/p/stop-words/downloads
    df = pd.read_fwf(PATH_DATASET + "stop-words-slovak.txt", encoding="utf-8")
    df2 = pd.read_fwf(PATH_DATASET + "stop-words-slovak.txt", encoding="utf-8")
    df = pd.concat([df, df2]).drop_duplicates()
    df.to_csv(r'faza1.csv')
    nlp = stanza.Pipeline('sk', verbose=False, processors="tokenize,lemma")
    annotated_stop_words = nlp(df.to_string(header=False, index=False))
    lemmatized_stopwords = []
    for word in annotated_stop_words.sentences[0].words:
        lemmatized_stopwords.append(word.lemma)
    df_lemmatized_stopwords = pd.DataFrame(lemmatized_stopwords, columns=["word"]).drop_duplicates()
    df_lemmatized_stopwords.to_csv(f'{PATH_DATASET}stop-words.txt', header=False, index=False)


def substitute(pattern, to, text):
    if type(pattern) is str:
        return re.sub(pattern, to, text, flags=re.ASCII)
    return text


def get_meta_by_pdfname(pdf_name):
    meta_info = pd.read_csv('../../Dataset/' + "all.csv", encoding="utf-8")
    pdf_name = int(re.findall("[0-9]+", pdf_name)[0])
    meta_info = meta_info.loc[meta_info['PDF'] == pdf_name]
    if meta_info.empty:
        print(f'Nemam {pdf_name}')
        return None
    meta_data = {}
    meta_data['kuv'] = meta_info['KUV'].values[0].split(' | ')
    meta_data['pvs'] = meta_info['Meno PVS'].values[0]
    meta_data['os'] = meta_info['Opravnena osoba'].values[0]
    meta_data['addr'] = meta_info['Adresa'].values[0]
    return meta_data


def translate_meta(meta_data):
    data = {
        'kuv': meta_data['KUV'].split(' | '),
        'pvs': meta_data['Obchodné meno'],
        'os': meta_data['os'],
        'addr': meta_data['Adresa sídla / miesto podnikania / bydliska']
    }
    return data


def replace_meta(text, meta_data):
    p = re.compile(r'([, ]+[sS]lovensk[aá] republika)')
    sk = re.search(p, meta_data['addr'])
    if sk is not None:
        meta_data['addr'] = meta_data['addr'][:sk.start()]
    for _kuv in meta_data['kuv']:
        text = substitute(_kuv, 'KUV', text)
        a = _kuv.split('. ')
        if len(a) > 1 and len(a[1]) > 3:
            text = substitute(a[1], 'KUV', text)
    text = substitute(meta_data['pvs'], 'PVS', text)
    text = substitute(meta_data['os'], 'OS', text)
    text = substitute(meta_data['addr'], 'ADDR', text)
    return text

class Classifier:

    def train(self, path_owners: str, path_managers: str, path_pretrained: str = None, save_model: bool = False):
        """Prepare model if it have one."""
        pass

    def is_owner(self, pdf_name: str) -> bool:
        """Method tries to extract information from pdf, if it describes KUV who is owner."""
        pass

    def is_owner(self, meta_data: list, pdf: fitz.Document):
        pass

    def get_stop_words(self):
        with open(self.path_to_dataset + "stop-words.txt", encoding="utf-8") as f:
            lines = f.readlines()
        stop_words = {lines[i].replace('\n', ''): i for i in range(len(lines))}
        return stop_words

    def write_desc(self, results):
        pass
