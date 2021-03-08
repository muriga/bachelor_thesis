import template
import stanza

class SimpleClassifier(template.Classifier):

    def get_stop_words(self):
        with open(self.path_to_dataset + "stop-words.txt", encoding="utf-8") as f:
            lines = f.readlines()
        stop_words = {i: lines[i].replace('\n','') for i in range(len(lines))}
        return stop_words

    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset
        self.stop_words = self.get_stop_words()
        self.nlp = stanza.Pipeline('sk', verbose=False,processors="tokenize,lemma")

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
        text = self.nlp(self.getText(path_to_pdf))
        print(text)
        return False

    def justTrying(self):
        #doc = nlp("Jeden priemerný text, ktorý by bolo dobré spracovať. Od jeho spracovania závisi odpoveď.")
        #print(doc)
        pass
