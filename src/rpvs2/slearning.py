from ocr import iterate_folder_get_text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor
import fitz
from sklearn.neural_network import MLPClassifier
from ocr import get_text
from utilities import replace_meta
import stanza
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

PATH_MODELS = "../../models/"
MAJITEL = 0
STATUTAR = 1


class SupervisedClassifier(Classifier):

    def __init__(self, path_dataset):
        self.path_to_dataset = path_dataset
        self.classifier = None
        self.nlp = stanza.Pipeline('sk', verbose=False, processors="tokenize")
        self.model_id = tm.strftime("%m-%d-%H%M%S")
        self.vectorizer = CountVectorizer(min_df=7, max_df=96, ngram_range=(1, 8), analyzer='char_wb')
        self.description = "CountVectorizer(min_df=7, max_df=96, ngram_range=(1, 8), analyzer='char_wb')" \
                           "not:TfidfTransformer(norm='l1', use_idf=True)\n" \
                           "MLPClassifier(solver='lbfgs', activation='relu', max_fun=7500, hidden_layer_sizes=(100, 50),\
                                        random_state=5, verbose=False, max_iter=400, n_iter_no_change=30, tol=0.001)"
        self.regex = PatternExtract(path_dataset)
        self.sentences_classifier = SentenceClassifier(path_dataset + "sentences.csv")
        self.bigram_vectorizer = CountVectorizer(min_df=5, ngram_range=(2, 2))

    # self.transformer = TfidfTransformer(norm='l1', use_idf=True)

    def train(self, path_owners: str, path_managers: str, path_pretrained: str = None, save_model: bool = False):
        if path_pretrained is not None:
            self.classifier = load(path_pretrained)
            texts, target, pdf_names = self.load_data(path_owners, path_managers)
            self.vectorizer.fit(texts)
            self.bigram_vectorizer.fit(texts)
            return
        texts, target, pdf_names = self.load_data(path_owners, path_managers)
        dt_matrix = self.vectorizer.fit_transform(texts).toarray()
        #dt_matrix = self.transformer.fit_transform(dt_matrix)

        bigrams = self.bigram_vectorizer.fit_transform(texts)
        dt_matrix = np.column_stack((dt_matrix, bigrams.toarray()))
        dt_matrix = self.add_sentences_features(dt_matrix, pdf_names)

        self.classifier = MLPClassifier(solver='lbfgs', activation='relu', hidden_layer_sizes=(100, 50),
                                        random_state=5, verbose=False).fit(dt_matrix, target)
        if save_model:
            dump(self.classifier, PATH_MODELS + "model_" + self.model_id + ".joblib")
        #print(self.vectorizer.get_feature_names())
        #print(self.bigram_vectorizer.get_feature_names())

    def add_sentences_features(self, dt_matrix, pdf_names):
        features = []
        #return dt_matrix
        for pdf in pdf_names:
            sentences = self.regex.get_sentences(pdf)
            pdf_feauters = {
                "0": 0,
                "1": 0,
                "2": 0
            }
            for sentence in sentences:
                predicted = self.sentences_classifier.predict(sentence)
                pdf_feauters[predicted] += 1
            features.append([list(pdf_feauters.values())[1] - list(pdf_feauters.values())[2]])
        arr = np.array(features)
        complete_features = np.column_stack((dt_matrix, arr))
        return complete_features

    def write_desc(self, results):
        text = "\t\tPrecision\tRecall\tF1score\nMajitel\t" + str(results['majitel']['precision']) + "\t" \
               + str(results['majitel']['recall']) + "\t\t" + str(results['majitel']['f1']) + "\n" \
               + "Statutar\t" + str(results['statutar']['precision']) + "\t" + str(results['statutar']['recall']) \
               + "\t" + str(results['statutar']['f1']) + "\n" + self.description
        with open(PATH_MODELS + "desc_" + self.model_id + ".txt", "w", encoding="utf-8") as file:
            file.write(text)

    def is_owner_testing(self, pdf_name: str, fact_is_owner) -> bool:
        if not pdf_name.endswith(".pdf"):
            pdf_name = pdf_name + ".pdf"
        text = get_text(self.path_to_dataset + 'test_all/' + pdf_name)
        if text is None:
            print(f'Cannot find {pdf_name}')
        # TODO get_metadata? replace metadata from list
        meta_data = get_meta_by_pdfname(pdf_name)
        text = replace_meta(text, meta_data)
        x = [text]
        x = self.vectorizer.transform(x).toarray()
        # x = self.transformer.transform(x)
        bigram = self.bigram_vectorizer.transform([text])
        x = np.column_stack((x, bigram.toarray()))
        x = self.add_sentences_features(x, [pdf_name])
        prediction = self.classifier.predict(x)
        if prediction == MAJITEL:
            return True  # self.regex.is_owner(pdf_name, False)
        return False

    def is_owner(self, meta_data: list, pdf: Union[str, fitz.Document]):
        meta_data = translate_meta(meta_data)
        text = ocr.convert_to_text(pdf)
        text = replace_meta(text, meta_data)
        x = [text]
        x = self.vectorizer.transform(x).toarray()
        #x = self.transformer.transform(x)
        x = self.add_sentences_features(x, [pdf])
        bigram = self.bigram_vectorizer.transform([text])
        x = np.column_stack((x, bigram.toarray()))
        prediction = self.classifier.predict(x)
        if prediction == MAJITEL:
            return True  # self.regex.is_owner(pdf_name, False)
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

    def tokenize(self, text):
        doc = self.nlp(text)
        tokenized = [token.text for sentence in doc.sentences for token in sentence.words]
        # print(tokenized)
        return tokenized


def load_testing(path_to_dataset):
    test_dict_data = iterate_folder_get_text(path_to_dataset + "test_majitel")
    for key in test_dict_data:
        test_dict_data[key] = replace_meta(test_dict_data[key], key)
    test_data = list(test_dict_data.values())

    test_dict_data = iterate_folder_get_text(path_to_dataset + "test_statutar")
    for key in test_dict_data:
        test_dict_data[key] = replace_meta(test_dict_data[key], key)
    test_data2 = list(test_dict_data.values())
    return test_data, test_data2


class SentenceClassifier():

    def __init__(self, path_cv):
        self.path_to_cv = path_cv
        self.regressor = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(90, 45), random_state=5, max_iter=50000)
        self.classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(90, 45), random_state=5, max_iter=50000)
        self.nlp = stanza.Pipeline('sk', verbose=False, processors="tokenize")
        self.model_id = tm.strftime("%m-%d-%H%M%S")
        stop_words = ["01", "aj", "11", "alebo", "by", "do", "že"]
        # self.vectorizer = CountVectorizer(min_df=0.03, max_df=0.95, ngram_range=(2, 3), stop_words=stop_words) #CountVectorizer(min_df=0.06, max_df=0.80, ngram_range=(1, 2))
        self.vectorizer = CountVectorizer(min_df=0.05, max_df=0.95, ngram_range=(1, 9), analyzer='char_wb')
        self.bigram_vectorizer = CountVectorizer(min_df=0.05, max_df=0.9, ngram_range=(2, 3))
        self.test = self.train()
        #self.train()
        self.evaluate()

    def train(self):
        majitel, statutar, null = self.load()
        # for i in range(len(majitel)):
        #    majitel[i][1] = 0
        training_set = majitel[:100]
        training_set = np.append(training_set, statutar[:100], axis=0)
        training_set = np.append(training_set, null[:300], axis=0)
        testing_set = majitel[100:]
        testing_set = np.append(testing_set, statutar[100:], axis=0)
        testing_set = np.append(testing_set, null[573:], axis=0) #538
        # np.random.set_state({"state": 5})
        # np.random.shuffle(training_set)
        x = training_set[0:, 0]
        y = training_set[0:, 1].astype('int')
        dt_matrix = self.vectorizer.fit_transform(x).toarray()
        bigrams = self.bigram_vectorizer.fit_transform(x).toarray()
        dt_matrix = np.column_stack((dt_matrix, bigrams))
        self.classifier.fit(dt_matrix, y)
        #self.regressor.fit(dt_matrix, y)

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
        x = self.vectorizer.transform([sentence]).toarray()
        bigrams = self.bigram_vectorizer.transform([sentence]).toarray()
        x = np.column_stack((x, bigrams))
        #print(self.regressor.predict(x)[0].astype('str').item())
        return self.classifier.predict(x)[0].astype('str').item()

    def evaluate(self):
        #print(self.vectorizer.get_feature_names())
        all_ans = {
            "0": 0,
            "1": 0,
            "2": 0
        }
        correct_ans = {
            "0": 0,
            "1": 0,
            "2": 0
        }
        all_majitel = 17
        all_statutar = 21
        all_null = 135-35

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
            # x = self.vectorizer.transform([i[0]])
            predicted = self.predict(i[0])[0]
            if predicted == str(i[1]):
                correct_ans[predicted] += 1
            else:
                print(f'predicted: {predicted} real: {str(i[1])}')
                print(i[0])
            ans[str(i[1])][predicted] += 1
            #else:
            #    if predicted != "0":
            #        print(f'{predicted} - {i[1]}')
            #        print(i[0])
            #    pass
            all_ans[predicted] += 1

        precision = correct_ans["1"] / all_ans["1"]
        recall = correct_ans["1"] / all_majitel
        f1 = (2 * precision * recall) / (precision + recall)
        print(f'Majitelia, precision:{precision}, recall:{recall}, f1:{f1}')
        precision = correct_ans["2"] / all_ans["2"]
        recall = correct_ans["2"] / all_statutar
        f1 = (2 * precision * recall) / (precision + recall)
        print(f'Statutari, precision:{precision}, recall:{recall}, f1:{f1}')
        precision = correct_ans["0"] / all_ans["0"]
        recall = correct_ans["0"] / all_null
        f1 = (2 * precision * recall) / (precision + recall)
        print(f'NULL, precision:{precision}, recall:{recall}, f1:{f1}')

        print(f'{ans["0"]["0"]}\t{ans["0"]["1"]}\t{ans["0"]["2"]}')
        print(f'{ans["1"]["0"]}\t{ans["1"]["1"]}\t{ans["1"]["2"]}')
        print(f'{ans["2"]["0"]}\t{ans["2"]["1"]}\t{ans["2"]["2"]}')
