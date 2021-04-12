import pandas as pd
from ocr import iterate_folder_get_text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
# TODO tr SGDCClassifier with Nystroem
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from ocr import get_text
from utilities import replace_meta
import re
import stanza
import pandas as pd
from template import Classifier
from joblib import dump, load
from pprint import pprint
import time as tm
from time import time
import numpy as np
import logging

PATH_MODELS = "../../models/"
MAJITEL = 0
STATUTAR = 1


class SupervisedClassifier(Classifier):

    def __init__(self, path_dataset):
        self.path_to_dataset = path_dataset
        self.classifier = None
        self.nlp = stanza.Pipeline('sk', verbose=False, processors="tokenize")
        self.model_id = tm.strftime("%m-%d-%H%M%S")
        self.vectorizer = None
        self.description = "CountVectorizer(min_df=13, max_df=100, ngram_range=(1,2)" \
                           "not:TfidfTransformer(norm='l1', use_idf=True)\n" \
                           "MLPClassifier(solver='lbfgs',activation='relu', max_fun=7500,hidden_layer_sizes=(100,45), random_state=5, verbose=False)"

    # self.transformer = TfidfTransformer(norm='l1', use_idf=True)

    def train(self, path_owners: str, path_managers: str, path_pretrained: str = None, save_model: bool = False):
        if path_pretrained is not None:
            self.classifier = load(path_pretrained)
            return
        texts, target = self.load_data(path_owners, path_managers)
        self.vectorizer = CountVectorizer(min_df=13, max_df=100, ngram_range=(1, 2))
        dt_matrix = self.vectorizer.fit_transform(texts)
        # dt_matrix = self.transformer.fit_transform(dt_matrix)
        self.classifier = MLPClassifier(solver='lbfgs', activation='relu', max_fun=7500, hidden_layer_sizes=(100, 45),
                                        random_state=5, verbose=False).fit(dt_matrix.toarray(), target)
        if save_model:
            dump(self.classifier, PATH_MODELS + "model_" + self.model_id + ".joblib")

    def find(self):
        texts, target = self.load_data(self.path_to_dataset + "majitel", self.path_to_dataset + "statutar")
        test_majitel, test_statutar = load_testing(self.path_to_dataset)
        texts.extend(test_majitel)
        texts.extend(test_statutar)
        target.extend([0 for i in range(len(test_majitel))])
        target.extend([1 for i in range(len(test_statutar))])
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MLPClassifier()),
        ])
        parameters = {
            # 'vect__min_df': (1, 2),
            'vect__min_df': (1, 2, 3, 4, 5, 8, 10, 13),
            'vect__max_df': (1.0, 0.5, 0.75, 0.9, 97, 98),
            'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
            'vect__tokenizer': (None, self.tokenize),
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__solver': ('lbfgs', 'adam'),
            'clf__alpha': (0.001, 0.0001, 0.005),
            'clf__learning_rate_init': (0.01, 0.005, 0.001, 0.0005),
            'clf__hidden_layer_sizes': ((100,), (100, 100), (100, 10))
        }
        grid_search = RandomizedSearchCV(pipeline, parameters, n_jobs=2, verbose=2)
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint(parameters)
        t0 = time()
        grid_search.fit(texts, target)
        print("done in %0.3fs" % (time() - t0))
        print()

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        df = pd.DataFrame(grid_search.cv_results_)
        df.to_csv("mlp.csv")

    def write_desc(self, results):
        text = "\t\tPrecision\tRecall\tF1score\nMajitel\t" + str(results['majitel']['precision']) + "\t" \
               + str(results['majitel']['recall']) + "\t\t" + str(results['majitel']['f1']) + "\n" \
               + "Statutar\t" + str(results['majitel']['precision']) + "\t" + str(results['majitel']['recall']) \
               + "\t" + str(results['majitel']['f1']) + "\n" + self.description
        with open(PATH_MODELS + "desc_" + self.model_id + ".txt", "w", encoding="utf-8") as file:
            file.write(text)

    def is_owner(self, pdf_name: str, fact_is_owner) -> bool:
        if not pdf_name.endswith(".pdf"):
            pdf_name = pdf_name + ".pdf"
        text = get_text(self.path_to_dataset + 'test/' + pdf_name)
        if text is None:
            print(f'Cannot find {pdf_name}, probably "majitel"')
            return True
        text = replace_meta(text, pdf_name)
        x = [text]
        x = self.vectorizer.transform(x)
        # x = self.transformer.transform(x)
        prediction = self.classifier.predict(x.toarray())
        if prediction == MAJITEL:
            return True
        return False

    def load_data(self, majitelia, statutari):
        dict_txts = iterate_folder_get_text(majitelia)
        dict_txts.update(iterate_folder_get_text(statutari))
        for key in dict_txts:
            dict_txts[key] = replace_meta(dict_txts[key], key)
        data = list(dict_txts.values())
        target = [0] * 50 + [1] * 50
        return data, target

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
        self.classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50,25), random_state=5, max_iter=50000)
        self.nlp = stanza.Pipeline('sk', verbose=False, processors="tokenize")
        self.model_id = tm.strftime("%m-%d-%H%M%S")
        stop_words = ["01", "aj", "11", "alebo", "by", "do", "že"]
        self.vectorizer = CountVectorizer(min_df=0.03, max_df=0.95, ngram_range=(2, 3), stop_words=stop_words) #CountVectorizer(min_df=0.06, max_df=0.80, ngram_range=(1, 2))
        self.description = "CountVectorizer(min_df=13, max_df=100, ngram_range=(1,2)" \
                           "not:TfidfTransformer(norm='l1', use_idf=True)\n" \
                           "MLPClassifier(solver='lbfgs',activation='relu', max_fun=7500,hidden_layer_sizes=(100,45), random_state=5, verbose=False)"
        self.test = self.train()
        self.evaluate()

    def train(self):
        majitel, statutar, null = self.load()
        training_set = majitel[:94]
        training_set = np.append(training_set, statutar[:96], axis=0)
        training_set = np.append(training_set, null[:538], axis=0)
        testing_set = majitel[94:]
        testing_set = np.append(testing_set, statutar[96:], axis=0)
        testing_set = np.append(testing_set, null[538:], axis=0)
        #np.random.set_state({"state": 5})
        np.random.shuffle(training_set)
        x = training_set[0:, 0]
        y = training_set[0:, 1].astype('int')
        x = self.vectorizer.fit_transform(x)
        self.classifier.fit(x, y)
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

    def evaluate(self):
        print(self.vectorizer.get_feature_names())
        all_ans = {
            "0" : 0,
            "1" : 0,
            "2" : 0
        }
        correct_ans = {
            "0" : 0,
            "1" : 0,
            "2" : 0
        }
        all_majitel = 23
        all_statutar = 25
        all_null = 135
        for i in self.test:
            x = self.vectorizer.transform([i[0]])
            predicted = self.classifier.predict(x)[0].astype('str').item()
            if predicted == str(i[1]):
                correct_ans[predicted] += 1
            else:
                if predicted != "0":
                    print(f'{predicted} - {i[1]}')
                    print(i[0])
                pass
            all_ans[predicted] += 1

        print(f'majitelia {correct_ans["1"]} z {all_ans["1"]}')
        print(f'statutari {correct_ans["2"]} z {all_ans["2"]}')
        print(f'null {correct_ans["0"]} z {all_ans["0"]}')
        precision = correct_ans["1"]/all_ans["1"]*100
        recall = correct_ans["1"]/all_majitel*100
        f1 = (2 * precision * recall) / (precision + recall)
        print(f'Majitelia, precision:{precision}, recall:{recall}, f1:{f1}')
        precision = correct_ans["2"]/all_ans["2"]*100
        recall = correct_ans["2"]/all_statutar*100
        f1 = (2 * precision * recall) / (precision + recall)
        print(f'Statutari, precision:{precision}, recall:{recall}, f1:{f1}')
        precision = correct_ans["0"]/all_ans["0"]*100
        recall = correct_ans["0"]/all_null*100
        f1 = (2 * precision * recall) / (precision + recall)
        print(f'Null, precision:{precision}, recall:{recall}, f1:{f1}')
