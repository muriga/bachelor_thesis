"""Module for different utilities for this project"""

import pandas as pd
import stanza
import re

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


def replace_meta(text, pdf_name):
    meta_info = pd.read_csv('../../Dataset/' + "all.csv", encoding="utf-8")
    pdf_name = int(re.findall("[0-9]+", pdf_name)[0])
    meta_info = meta_info.loc[meta_info['PDF'] == pdf_name]
    if meta_info.empty:
        print(f'Nemam {pdf_name}')
        return text
    _KUV = meta_info['KUV'].values[0].split(' | ')
    # TODO Titul. meno priezvisko: treba sparsovat nech nahradi aj bez titulu
    _PVS = meta_info['Meno PVS'].values[0]
    _OS = meta_info['Opravnena osoba'].values[0]
    _ADDR = meta_info['Adresa'].values[0]
    p = re.compile(r'([, ]+[sS]lovensk[aá] republika)')
    sk = re.search(p, _ADDR)
    if sk is not None:
        _ADDR = _ADDR[:sk.start()]
    for _kuv in _KUV:
        text = substitute(_kuv, 'KUV', text)
    text = substitute(_PVS, 'PVS', text)
    text = substitute(_OS, 'OS', text)
    text = substitute(_ADDR, 'ADDR', text)
    return text


