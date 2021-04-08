import pandas as pd
from ocr import iterate_folder_get_text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from ocr import get_text
from utilities import replace_meta
import re
import stanza
import pandas as pd
from template import Classifier
from joblib import dump, load
import time

PATH_MODELS = "../../models/"
MAJITEL = 0
STATUTAR = 1





class SupervisedClassifier(Classifier):

    def __init__(self, path_dataset):
        self.path_to_dataset = path_dataset
        self.classifier = None
        self.nlp = stanza.Pipeline('sk', verbose=False, processors="tokenize,lemma")
        self.model_id = time.strftime("%m-%d-%H%M%S")
        self.vectorizer = None
        self.description = "CountVectorizer(min_df=3, max_df=97)\n" \
                           "MLPClassifier(solver='lbfgs', random_state=5, verbose=False)"

    def train(self, path_owners: str, path_managers: str, path_pretrained: str = None, save_model: bool = False):
        if path_pretrained is not None:
            self.classifier = load(path_pretrained)
            return
        texts, target = self.load_data(path_owners, path_managers)
        self.vectorizer = CountVectorizer(min_df=3, max_df=97)
        dt_matrix = self.vectorizer.fit_transform(texts)
        self.classifier = MLPClassifier(solver='lbfgs', random_state=5, verbose=False).fit(dt_matrix.toarray(), target)
        if save_model:
            dump(self.classifier, PATH_MODELS + "model_" + self.model_id + ".joblib")
            with open(PATH_MODELS + "desc_" + self.model_id + ".txt", "w", encoding="utf-8") as file:
                file.write(self.description)

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
