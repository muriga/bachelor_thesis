from ocr import iterate_folder_get_text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor
import fitz
from sklearn.neural_network import MLPClassifier
from ocr import get_text
from utilities import replace_meta
import pandas as pd
from template import Classifier
from joblib import dump, load
import time as tm
import numpy as np
import ocr
from pattern import PatternExtract
from utilities import get_meta_by_pdfname
from utilities import translate_meta
from typing import Union
import re

PATH_MODELS = "../../models/"
MAJITEL = 0
STATUTAR = 1


def load_data(majitelia, statutari):
    documents_majitelia = iterate_folder_get_text(majitelia)
    documents_statutari = iterate_folder_get_text(statutari)
    documents = documents_majitelia | documents_statutari
    for key in documents:
        meta_data = get_meta_by_pdfname(key)
        documents[key] = replace_meta(documents[key], meta_data)
    data = list(documents.values())
    target = [0] * 50 + [1] * 50
    return data, target, [*documents]


class MLPClassifierBoW(Classifier):

    def __init__(self, path_dataset):
        self.path_to_dataset = path_dataset
        self.classifier = None
        self.model_id = tm.strftime("%m-%d-%H%M%S")
        self.vectorizer = CountVectorizer(min_df=7, max_df=96, ngram_range=(1, 8), analyzer='char_wb')
        self.description = "CountVectorizer(min_df=7, max_df=96, ngram_range=(1, 8), analyzer='char_wb')" \
                           "not:TfidfTransformer(norm='l1', use_idf=True)\n" \
                           "MLPClassifier(solver='lbfgs', activation='relu', max_fun=7500, hidden_layer_sizes=(100, 50),\
                                        random_state=5, verbose=False, max_iter=400, n_iter_no_change=30, tol=0.001)"
        self.bigram_vectorizer = CountVectorizer(min_df=5, ngram_range=(2, 2))

    def train(self, path_owners: str, path_managers: str, path_pretrained: str = None, save_model: bool = False):
        if path_pretrained is not None:
            self.classifier = load(path_pretrained)
            texts, target, pdf_names = load_data(path_owners, path_managers)
            self.vectorizer.fit(texts)
            self.bigram_vectorizer.fit(texts)
            return
        texts, target, pdf_names = load_data(path_owners, path_managers)
        dt_matrix_ngram_chars = self.vectorizer.fit_transform(texts).toarray()
        dt_matrix_bigram_words = self.bigram_vectorizer.fit_transform(texts).toarray()
        dt_matrix = np.column_stack((dt_matrix_ngram_chars, dt_matrix_bigram_words))

        self.classifier = MLPClassifier(solver='lbfgs', activation='relu', hidden_layer_sizes=(100, 50),
                                        random_state=5, verbose=False).fit(dt_matrix, target)
        if save_model:
            dump(self.classifier, PATH_MODELS + "model_" + self.model_id + ".joblib")

    def is_owner(self, meta_data: list, pdf: Union[str, fitz.Document]):
        if 'KUV' in meta_data:
            meta_data = translate_meta(meta_data)
            text = ocr.convert_to_text(pdf)
        else:
            text = get_text(self.path_to_dataset + 'test_all/' + pdf + ".pdf")
        text = replace_meta(text, meta_data)
        vector_ngram_chars = self.vectorizer.transform([text]).toarray()
        vector_bigram_words = self.bigram_vectorizer.transform([text])
        document_vector = np.column_stack((vector_ngram_chars, vector_bigram_words.toarray()))
        prediction = self.classifier.predict(document_vector)
        if prediction == MAJITEL:
            return True
        return False

    def write_desc(self, results):
        text = "\t\tPrecision\tRecall\tF1score\nMajitel\t" + str(results['majitel']['precision']) + "\t" \
               + str(results['majitel']['recall']) + "\t\t" + str(results['majitel']['f1']) + "\n" \
               + "Statutar\t" + str(results['statutar']['precision']) + "\t" + str(results['statutar']['recall']) \
               + "\t" + str(results['statutar']['f1']) + "\n" + self.description
        with open(PATH_MODELS + "desc_" + self.model_id + ".txt", "w", encoding="utf-8") as file:
            file.write(text)


class MLPClassifierWSent(Classifier):
    def __init__(self, path_dataset):
        self.path_to_dataset = path_dataset
        self.classifier = None
        self.model_id = tm.strftime("%m-%d-%H%M%S")
        self.vectorizer = CountVectorizer(min_df=7, max_df=96, ngram_range=(1, 8), analyzer='char_wb')
        self.description = "CountVectorizer(min_df=7, max_df=96, ngram_range=(1, 8), analyzer='char_wb')" \
                           "not:TfidfTransformer(norm='l1', use_idf=True)\n" \
                           "MLPClassifier(solver='lbfgs', max_fun=7500, hidden_layer_sizes=(100, 50),\
                                        random_state=5, verbose=False, max_iter=400, n_iter_no_change=30, tol=0.001)"
        self.regex = PatternExtract(path_dataset)
        self.sentences_classifier = SentenceClassifier(path_dataset + "sentences.csv")
        self.bigram_vectorizer = CountVectorizer(min_df=5, ngram_range=(2, 2))

    def train(self, path_owners: str, path_managers: str, path_pretrained: str = None, save_model: bool = False):
        if path_pretrained is not None:
            self.classifier = load(path_pretrained)
            texts, target, pdf_names = self.load_data(path_owners, path_managers)
            self.vectorizer.fit(texts)
            self.bigram_vectorizer.fit(texts)
            return
        texts, target, pdf_names = self.load_data(path_owners, path_managers)
        dt_matrix_ngram_chars = self.vectorizer.fit_transform(texts).toarray()
        dt_matrix_bigram_words = self.bigram_vectorizer.fit_transform(texts).toarray()
        dt_matrix = np.column_stack((dt_matrix_ngram_chars, dt_matrix_bigram_words))
        dt_matrix = self.add_sentences_features(dt_matrix, pdf_names)

        self.classifier = MLPClassifier(solver='lbfgs', activation='relu', hidden_layer_sizes=(100, 50),
                                        random_state=5, verbose=False).fit(dt_matrix, target)
        if save_model:
            dump(self.classifier, PATH_MODELS + "model_" + self.model_id + ".joblib")

    def add_sentences_features(self, dt_matrix, pdf_names):
        count_of_type_diff = []
        for pdf in pdf_names:
            sentences = self.get_sentences(pdf)
            types = {
                "0": 0,
                "1": 0,
                "2": 0
            }
            for sentence in sentences:
                predicted = self.sentences_classifier.predict(sentence)
                types[predicted] += 1
            count_of_type_diff.append([list(types.values())[1] - list(types.values())[2]])
        diff_as_np_array = np.array(count_of_type_diff)
        complete_features = np.column_stack((dt_matrix, diff_as_np_array))
        return complete_features

    def get_sentences(self, pdf_name):
        if not pdf_name.endswith(".pdf"):
            pdf_name = pdf_name + ".pdf"
        text = get_text(self.path_to_dataset + 'test_all/' + pdf_name)
        text = self.regex.replace_meta(text, pdf_name)
        text = self.regex.preprocessing(text)
        patterns = ["([^.]| z \\.| [0-9$§]+ . | p.sm \\. | ods \\.)*pvs([^.]| z \\.|[0-9$§]+ \\. )*kuv([^.]| "
                    "z \\.| [0-9$§]+ . | p.sm \\. | ods \\.)*",
                    "([^.]| z \\.| [0-9$§]+ . | p.sm \\. | ods \\.)*kuv([^.]| z \\.|[0-9$§]+ \\. )*pvs([^.]| "
                    "z \\.| [0-9$§]+ . | p.sm \\. | ods \\.)*"]
        sentences = self.do_get_sentences(patterns, text)
        return sentences

    def do_get_sentences(self, patterns, text):
        sentences = []
        for pattern in patterns:
            found = re.finditer(pattern, text, flags=re.UNICODE)
            sentences = self.append(sentences, found)
        return sentences

    def append(self, sentences, found):
        for i in found:
            sentences.append(i.group())
        return sentences

    def write_desc(self, results):
        text = "\t\tPrecision\tRecall\tF1score\nMajitel\t" + str(results['majitel']['precision']) + "\t" \
               + str(results['majitel']['recall']) + "\t\t" + str(results['majitel']['f1']) + "\n" \
               + "Statutar\t" + str(results['statutar']['precision']) + "\t" + str(results['statutar']['recall']) \
               + "\t" + str(results['statutar']['f1']) + "\n" + self.description
        with open(PATH_MODELS + "desc_" + self.model_id + ".txt", "w", encoding="utf-8") as file:
            file.write(text)

    def is_owner(self, meta_data: list, pdf: Union[str, fitz.Document]):
        if 'KUV' in meta_data:
            meta_data = translate_meta(meta_data)
            text = ocr.convert_to_text(pdf)
        else:
            text = get_text(self.path_to_dataset + 'test_all/' + pdf + ".pdf")
        text = replace_meta(text, meta_data)
        vector_ngram_chars = self.vectorizer.transform([text]).toarray()
        vector_bigram_words = self.bigram_vectorizer.transform([text])
        document_vector = np.column_stack((vector_ngram_chars, vector_bigram_words.toarray()))
        document_vector = self.add_sentences_features(document_vector, [pdf])

        prediction = self.classifier.predict(document_vector)
        if prediction == MAJITEL:
            return True
        return False

    def load_data(self, majitelia, statutari):
        dict_txts = iterate_folder_get_text(majitelia)
        dict_txts.update(iterate_folder_get_text(statutari))
        for key in dict_txts:
            meta_data = get_meta_by_pdfname(key)
            dict_txts[key] = replace_meta(dict_txts[key], meta_data)
        data = list(dict_txts.values())
        target = [0] * 50 + [1] * 50
        return data, target, [*dict_txts]


class SentenceClassifier():

    def __init__(self, path_cv):
        self.path_to_cv = path_cv
        self.regressor = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(90, 45), random_state=5, max_iter=50000)
        self.classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(90, 45), random_state=5, max_iter=50000)
        self.model_id = tm.strftime("%m-%d-%H%M%S")
        self.vectorizer = CountVectorizer(min_df=0.05, max_df=0.95, ngram_range=(1, 9), analyzer='char_wb')
        self.bigram_vectorizer = CountVectorizer(min_df=0.05, max_df=0.9, ngram_range=(2, 3))
        self.train()

    def train(self):
        majitel, statutar, null = self.load()
        training_set = majitel[:100]
        training_set = np.append(training_set, statutar[:100], axis=0)
        training_set = np.append(training_set, null[:300], axis=0)

        testing_set = majitel[100:]
        testing_set = np.append(testing_set, statutar[100:], axis=0)
        testing_set = np.append(testing_set, null[573:], axis=0)

        sentences = training_set[0:, 0]
        classes = training_set[0:, 1].astype('int')

        dt_matrix_ngrams_chars = self.vectorizer.fit_transform(sentences).toarray()
        dt_matrix_bigrams_words = self.bigram_vectorizer.fit_transform(sentences).toarray()
        dt_matrix = np.column_stack((dt_matrix_ngrams_chars, dt_matrix_bigrams_words))
        self.classifier.fit(dt_matrix, classes)
        return testing_set

    def load(self):
        data_frame = pd.read_csv(self.path_to_cv)
        statutar = data_frame.loc[data_frame["Label"] == 2]
        statutar = statutar[["Sentence", "Label"]].drop_duplicates().to_numpy()
        majitel = data_frame.loc[data_frame["Label"] == 1]
        majitel = majitel[["Sentence", "Label"]].drop_duplicates().to_numpy()
        null = data_frame.loc[data_frame["Label"] == 0]
        null = null[["Sentence", "Label"]].drop_duplicates().to_numpy()
        return majitel, statutar, null

    def predict(self, sentence):
        vector_ngrams_chars = self.vectorizer.transform([sentence]).toarray()
        vector_bigram_words = self.bigram_vectorizer.transform([sentence]).toarray()
        sentence_vector = np.column_stack((vector_ngrams_chars, vector_bigram_words))
        return self.classifier.predict(sentence_vector)[0].astype('str').item()

    def evaluate(self):
        ans = {
            "0": {"0": 0,
                  "1": 0,
                  "2": 0},
            "1": {"0": 0,
                  "1": 0,
                  "2": 0},
            "2": {"0": 0,
                  "1": 0,
                  "2": 0}
        }
        for i in self.test:
            predicted = self.predict(i[0])[0]
            ans[str(i[1])][predicted] += 1

        print(f'{ans["0"]["0"]}\t{ans["0"]["1"]}\t{ans["0"]["2"]}')
        print(f'{ans["1"]["0"]}\t{ans["1"]["1"]}\t{ans["1"]["2"]}')
        print(f'{ans["2"]["0"]}\t{ans["2"]["1"]}\t{ans["2"]["2"]}')
