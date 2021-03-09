import template
import stanza
import pandas as pd

class SimpleClassifier(template.Classifier):

    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset
        self.stop_words = self.get_stop_words()
        self.nlp = stanza.Pipeline('sk', verbose=False,processors="tokenize,lemma")

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

    def is_owner(self, path_to_pdf:str) -> bool:
        if path_to_pdf.endswith(".pdf"):
            path_to_pdf.removesuffix(".pdf")
        path_to_pdf = path_to_pdf + '.txt'
        raw_text = self.getText(path_to_pdf)
        text = self.nlp(raw_text)
        text = self.remove_stopwords(text)
        return False

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