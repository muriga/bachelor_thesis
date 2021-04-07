import pandas as pd
from ocr import iterate_folder_get_text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import metrics
from ocr import get_text
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
            dump(self.classifier ,PATH_MODELS + "model_" + self.model_id + ".joblib")
            with open(PATH_MODELS + "desc_" + self.model_id + ".txt", "w", encoding="utf-8") as file:
                file.write(self.description)



    def is_owner(self, pdf_name: str, fact_is_owner) -> bool:
        owner = True
        if not pdf_name.endswith(".pdf"):
            pdf_name = pdf_name + ".pdf"
        text = get_text(self.path_to_dataset + 'test2/' + pdf_name)
        if text is None:
            print(f'Cannot find {pdf_name}, probably "majitel"')
            return True
        text = self.replace_meta(text, pdf_name)
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
            dict_txts[key] = self.replace_meta(dict_txts[key], key)
        data = list(dict_txts.values())
        target = [0] * 50 + [1] * 50
        return data, target

    def replace_meta(self, text, pdf_name):
        meta_info = pd.read_csv('../../Dataset/' + "all.csv", encoding="utf-8")
        pdf_name = int(re.findall("[0-9]+", pdf_name)[0])
        meta_info = meta_info.loc[meta_info['PDF'] == pdf_name]
        if meta_info.empty:
            print(f'Nemam {pdf_name}')
            return text  # TODO chyba -> niektori z ulozenych sa medzitym vymazali
        _KUV = meta_info['KUV'].values[0].split(' | ')
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


class Dataset:
    def __init__(self):
        self.target  # list containing 0 for majitel and 1 for statutar


def load_learning(path_to_dataset):
    dict_data = iterate_folder_get_text(path_to_dataset + "majitel")
    dict_data.update(iterate_folder_get_text(path_to_dataset + "statutar"))
    for key in dict_data:
        dict_data[key] = replace_meta(dict_data[key], key)
    data = list(dict_data.values())
    data = preprocessing(data)
    target = []
    for i in range(50):
        target.append(0)
    for i in range(50):
        target.append(1)
    return data, target


def load_testing(path_to_dataset):
    test_dict_data = iterate_folder_get_text(path_to_dataset + "test_majitel")
    for key in test_dict_data:
        test_dict_data[key] = replace_meta(test_dict_data[key], key)
    test_data = list(test_dict_data.values())
    test_data = preprocessing(test_data)

    test_dict_data = iterate_folder_get_text(path_to_dataset + "test_statutar")
    for key in test_dict_data:
        test_dict_data[key] = replace_meta(test_dict_data[key], key)
    test_data2 = list(test_dict_data.values())
    test_data2 = preprocessing(test_data2)
    return test_data, test_data2


def find_model(path_to_dataset):
    data, target = load_learning(path_to_dataset)
    test_data, test_data2 = load_testing(path_to_dataset)
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)),
    ])
    ngram_vectorizer = CountVectorizer(min_df=3, max_df=97, strip_accents='unicode', encoding='utf-8',
                                       ngram_range=(1, 1))
    X = ngram_vectorizer.fit_transform(data)
    clf = MLPClassifier(solver='lbfgs', random_state=5, verbose=False).fit(X.toarray(), target)
    # clf = SVC(kernel='sigmoid').fit(X.toarray(), target)
    X_test = ngram_vectorizer.transform(test_data)
    predicted_majitel = clf.predict(X_test.toarray())
    X_test = ngram_vectorizer.transform(test_data2)
    predicted_statutar = clf.predict(X_test.toarray())

    # gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
    # gs_clf.fit(data, target)

    # predicted_majitel = gs_clf.predict(test_data)
    print(predicted_majitel)
    # predicted_statutar = gs_clf.predict(test_data2)
    print(predicted_statutar)
    nespravne_majitel = 0
    spravne_statutar = 0
    for i in range(len(predicted_majitel)):
        if predicted_majitel[i] == 1:
            nespravne_majitel += 1
    for i in range(len(predicted_statutar)):
        if predicted_statutar[i] == 1:
            spravne_statutar += 1
    precision = spravne_statutar / (spravne_statutar + nespravne_majitel) * 100
    recall = spravne_statutar / len(predicted_statutar) * 100
    f_score = (2 * precision * recall) / (precision + recall)
    print(f'Precision is {precision}')
    print(f'Recall is {recall}')
    print(f'Recall is {f_score}')
    print(clf.get_params())
    # print(gs_clf.best_score_)
    # for param_name in sorted(parameters.keys()):
    #    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def get_matrix2(path_to_dataset):
    data, target = load_learning(path_to_dataset)
    test_data, test_data2 = load_testing(path_to_dataset)

    text_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)),
        ('tfdif', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=5, max_iter=5, tol=None)),
    ])
    text_clf.fit(data, target)
    predicted_majitel = text_clf.predict(test_data)
    print(predicted_majitel)
    predicted_statutar = text_clf.predict(test_data2)
    print(predicted_statutar)
    nespravne_majitel = 0
    spravne_statutar = 0
    for i in range(len(predicted_majitel)):
        if predicted_majitel[i] == 1:
            nespravne_majitel += 1
    for i in range(len(predicted_statutar)):
        if predicted_statutar[i] == 1:
            spravne_statutar += 1
    print(f'Precision is {spravne_statutar / (spravne_statutar + nespravne_majitel) * 100}')
    print(f'Recall is {spravne_statutar / len(predicted_statutar) * 100}')


def get_matrix(majitel):
    dict_data = iterate_folder_get_text(majitel)
    data = list(dict_data.values())
    data = preprocessing(data)
    # print(len(list(data.values())))
    vectorizer = CountVectorizer()
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)
    X_2 = ngram_vectorizer.fit_transform(data)
    transformer = TfidfTransformer(smooth_idf=True)
    tfdif = transformer.fit_transform(X_2)
    # print(tfdif.toarray())
    # print(len(tfdif.toarray()[0]))
    # df = pd.DataFrame(tfdif.toarray(), index=dict_data.keys(),columns=ngram_vectorizer.get_feature_names())
    # df.to_csv('tfdif.csv',encoding='utf-8')
    # print((ngram_vectorizer.get_feature_names()))
    # print(vectorizer.get_feature_names())


def preprocessing(data):
    vm_kuv = "((jeho|jej)( |\n)*)?(.len([^ \n]*)?( |\n)*)?((jeho|jej)( |\n)*)?(vrcholov[^ \n]*" \
             "( |\n)*mana.ment[^ \n]*|(predstav[^ \n]*( |\n)*)?.tatut.r[^ \n]*( |\n)*org.n[^ \n]*)+"
    pvs = "[Pp]artner([^ \n]*)?(( |\n)*verejn.([^ \n]*)?( |\n)*sektora)?"
    kuv = "[Kk]one.n[^ \n]*( |\n)*u..vate[^ \n]*( |\n)*v.hod[^ \n]*"
    nofo = "((.iadn[^ \n]*)( |\n)*)+fyzick.( |\n)*osob."
    fo = "fyzick.( |\n)*osob."
    os = ".pr.vn[^ \n]*( |\n)*.sob[^ \n]*"  # oproti pattern maly rozdiel v zakx
    zakx = "[8$§](\.| |\n)*6a(\.| |\n)*(od)?([^ \n])*(\.| |\n)*([1-9I]?(\.| |\n)*)((p.s([^ \n])*" \
           "(\.| |\n)*([a-z)])*)?( |\n)*)?[Zz].ko([^ \n])*(\.| |\n)*(.\.(\.| |\n)*297\/2008(\.| |\n)*" \
           ".(\.| |\n)*.(\.| |\n)*)?(o( |\n))?(ochrane( |\n)*)?(pred( |\n)*)?(leg([^ \n])*( |\n)*)?" \
           "(pr.j([^ \n])*( |\n)*)?(z( |\n)*)?(tr([^ \n])*( |\n)*)?(.inn([^ \n])*(\n| ))?(a )?(o )?" \
           "(ochrane (pred )?fin([^ \n])* ter([^ \n])* . . zmene a do([^ \n])* nie([^ \n])*( |\n)*" \
           "z.k([^ \n])*( |\n))?((v )?znen([^ \n])*( |\n)nesk([^ \n])*( |\n)pred([^ \n])*)?"
    pvs2 = "[pP][VvY][S5]"
    for i in range(len(data)):
        # TODO Napojit na NER? Alebo take minimum: prejdi aspon mena a priezviska KUV, ci sa z nich nieco nenajde
        # data[i] = substitute(vm_kuv, "vmkuv", data[i])
        # data[i] = substitute(pvs, "PVS", data[i])
        # data[i] = substitute(kuv, "kuv", data[i])
        # data[i] = substitute(nofo, "nofo", data[i])
        # data[i] = substitute(fo, "fo", data[i])
        # data[i] = substitute(os, "os", data[i])
        # data[i] = substitute("sp..a", "spĺňa", data[i])
        # data[i] = substitute("KÚV", "kuv", data[i])
        # data[i] = substitute(zakx, "zak1", data[i])
        # data[i] = tokenize(data[i])
        # data[i] = substitute(pvs2, "PVS", data[i])
        pass
    return data


def substitute(pattern, to, text):
    if type(pattern) is str:
        return re.sub(pattern, to, text)
    return text


def tokenize(text):
    tokenized = ""
    doc = nlp(text)
    # for i, sentence in enumerate(doc.sentences):
    for sentence in doc.sentences:
        for token in sentence.words:
            if token.text[:2] == "ne" and token.lemma[:2] != "ne":
                tokenized += "ne" + token.lemma + " "
            else:
                tokenized += token.lemma + " "
    return tokenized


def replace_meta(text, pdf_name):
    meta_info = pd.read_csv('../../Dataset/' + "all.csv", encoding="utf-8")
    pdf_name = int(re.findall("[0-9]+", pdf_name)[0])
    meta_info = meta_info.loc[meta_info['PDF'] == pdf_name]
    if meta_info.empty:
        # print(f'Nemam {pdf_name}')
        return text  # TODO chyba -> niektori z ulozenych sa medzitym vymazali
        # raise Exception(f'Unable to use metainformations - I can\'t find PVS with name of pdf: \'{pdf_name}\'')
    _KUV = meta_info['KUV'].values[0].split(' | ')
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
