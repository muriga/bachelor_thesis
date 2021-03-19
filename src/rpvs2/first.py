import template
import stanza
import pandas as pd
import re


def substitute(pattern, to, text):
    if type(pattern) is str:
        return re.sub(pattern, to, text, flags=re.ASCII)
    return text


class SimpleClassifier(template.Classifier):

    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset
        self.stop_words = self.get_stop_words()
        self.nlp = stanza.Pipeline('sk', verbose=False,processors="tokenize,lemma")
        self.meta_info = pd.read_csv(path_to_dataset + "all.csv", encoding="utf-8")

    def get_stop_words(self):
        with open(self.path_to_dataset + "stop-words.txt", encoding="utf-8") as f:
            lines = f.readlines()
        stop_words = {lines[i].replace('\n',''):i for i in range(len(lines))}
        return stop_words

    def getText(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        return text


    def train(self, path_to_owners: str, path_to_managers: str):
        print("Trenujem poctivo")

    def is_owner(self, pdf_name:str) -> bool:
        #print(pdf_name)
        if pdf_name.endswith(".pdf"):
            pdf_name = pdf_name.removesuffix(".pdf")
        path_to_pdf = self.path_to_dataset + 'test1/' + pdf_name + '.txt'
        raw_text = self.getText(path_to_pdf)
        raw_text = self.replace_meta(raw_text, pdf_name)
        if self.test_re(raw_text):
            return False
        #text = self.nlp(raw_text)
        #text = self.remove_stopwords(text)
        return True

    def justTrying(self):
        #doc = nlp("Jeden priemerný text, ktorý by bolo dobré spracovať. Od jeho spracovania závisi odpoveď.")
        #print(doc)
        pass

    def remove_stopwords(self, text):
        text_without_stopwords = []
        for sentence in text.sentences:
            for word in sentence.words:
                if word.lemma not in self.stop_words:
                    text_without_stopwords.append(word.lemma)
        df_text_without_stopwords = pd.DataFrame(text_without_stopwords, columns=['word'])
        return df_text_without_stopwords

    def replace_meta(self, text, path):
        pdf_name = int(re.findall("[0-9]+",path)[0])
        meta_info = self.meta_info.loc[self.meta_info['PDF'] == pdf_name]
        if meta_info.empty:
            return text #TODO chyba -> niektori z ulozenych sa medzitym vymazali
            #raise Exception(f'Unable to use metainformations - I can\'t find PVS with name of pdf: \'{pdf_name}\'')
        _KUV = meta_info['KUV'].values[0]
        _PVS = meta_info['Meno PVS'].values[0]
        _OS = meta_info['Opravnena osoba'].values[0]
        _ADDR = meta_info['Adresa'].values[0]
        p = re.compile(r'([, ]+[sS]lovensk[aá] republika)')
        sk = re.search(p, _ADDR)
        if sk is not None:
            _ADDR = _ADDR[:sk.start()]

        text = substitute(_KUV,'KUV', text)
        text = substitute(_PVS,'PVS', text)
        text = substitute(_OS,'OS', text)
        text = substitute(_ADDR,'ADDR', text)
        return text

    def first_pattern(selfs, text):
        pattern = re.search()

    def test_re(self, text):
        first_attempt = re.search('ako členovia', text)
        if first_attempt is not None:
            return True
        pattern = r'z[aá]pis [cč]lenov vrcholov[eé]ho mana[zž]mentu([^A-W ]*)s[úu] splnen[eé]'
        second_attempt = re.search(pattern, text)
        if second_attempt is not None:
            return True
        pattern2 = r'členovia vrcholového'
        third_attempt = re.search(pattern2, text)
        if third_attempt is not None:
            return True
        return False