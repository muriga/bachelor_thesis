import pandas as pd
from ocr import iterate_folder_get_text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
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
        self.description = "CountVectorizer(min_df=13, max_df=100, ngram_range=(1,2))\n" \
                           "not:TfidfTransformer(norm='l1', use_idf=True)\n" \
                           "MLPClassifier(solver='lbfgs',activation='relu', max_fun=7500,hidden_layer_sizes=(100,45), random_state=5, verbose=False)"
       # self.transformer = TfidfTransformer(norm='l1', use_idf=True)

    def train(self, path_owners: str, path_managers: str, path_pretrained: str = None, save_model: bool = False):
        if path_pretrained is not None:
            self.classifier = load(path_pretrained)
            return
        texts, target = self.load_data(path_owners, path_managers)
        self.vectorizer = CountVectorizer(min_df=13, max_df=100, ngram_range=(1,2))
        dt_matrix = self.vectorizer.fit_transform(texts)
        #dt_matrix = self.transformer.fit_transform(dt_matrix)
        self.classifier = MLPClassifier(solver='lbfgs',activation='relu', max_fun=7500, hidden_layer_sizes=(100,45), random_state=5, verbose=False).fit(dt_matrix.toarray(), target)
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
            #'vect__min_df': (1, 2),
            'vect__min_df': (1, 2, 3, 4, 5, 8, 10, 13),
            'vect__max_df': (1.0, 0.5, 0.75, 0.9, 97, 98),
            'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
            'vect__tokenizer': (None, self.tokenize),
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__solver': ('lbfgs', 'adam'),
            'clf__alpha': (0.001, 0.0001, 0.005),
            'clf__learning_rate_init': (0.01, 0.005, 0.001, 0.0005),
            'clf__hidden_layer_sizes': ((100,), (100,100), (100,10))
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
               + "Statutar\t" +  str(results['majitel']['precision']) + "\t" + str(results['majitel']['recall']) \
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
        #x = self.transformer.transform(x)
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
