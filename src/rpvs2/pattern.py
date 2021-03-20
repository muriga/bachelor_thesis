from template import Classifier
import stanza
import pandas as pd
import re


class PatternExtract(Classifier):
    """Class is providing evaluation if given pdf file is 'statutar'. For this purpose it use patterns"""

    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset
        self.stop_words = super().get_stop_words()
        self.nlp = stanza.Pipeline('sk', verbose=False, processors="tokenize,lemma")
        self.meta_info = pd.read_csv(path_to_dataset + "all.csv", encoding="utf-8")
        self.patterns = self.get_setting_patterns()

    def replace_meta(self, text, path):
        pdf_name = int(re.findall("[0-9]+", path)[0])
        meta_info = self.meta_info.loc[self.meta_info['PDF'] == pdf_name]
        if meta_info.empty:
            return text  # TODO chyba -> niektori z ulozenych sa medzitym vymazali
            # raise Exception(f'Unable to use metainformations - I can\'t find PVS with name of pdf: \'{pdf_name}\'')
        _KUV = meta_info['KUV'].values[0]
        _PVS = meta_info['Meno PVS'].values[0]
        _OS = meta_info['Opravnena osoba'].values[0]
        _ADDR = meta_info['Adresa'].values[0]
        p = re.compile(r'([, ]+[sS]lovensk[aá] republika)')
        sk = re.search(p, _ADDR)
        if sk is not None:
            _ADDR = _ADDR[:sk.start()]

        text = self.substitute(_KUV, 'KUV', text)
        text = self.substitute(_PVS, 'PVS', text)
        text = self.substitute(_OS, 'OS', text)
        text = self.substitute(_ADDR, 'ADDR', text)
        return text

    def substitute(self, pattern, to, text):
        if type(pattern) is str:
            return re.sub(pattern, to, text, flags=re.ASCII)
        return text

    def itarate_patterns(self):
        tokenized_patterns = []
        for p in self.patterns:
            pattern = ""
            doc = self.nlp(p)
            #for i, sentence in enumerate(doc.sentences):
            for sentence in doc.sentences:
                for token in sentence.words:
                    #only if is not in stop words?
                    if token.lemma not in self.stop_words:
                        pattern += token.lemma + " "
                    else:
                        print(token.lemma)
            tokenized_patterns.append(pattern)
        for i in range(len(tokenized_patterns)):
            print(f'{i}. pattern: {tokenized_patterns[i]}')

    def get_patterns(self):
        patterns = []
        patterns.append(re.compile(r'podmienka zápis členov vrcholového manažment '
                       r'ust \. [$8§] 4 ods[a-zA-Z0-9.,§$ ]{0,80} sú splnené')) #0.
        patterns.append(re.compile("spoločnosť (nepriamo )*ovládanej emitentom cenných papier")) #1
        patterns.append("")


    def get_setting_patterns(self):
        #chcelo by to predtym zmenit veci typu konečný užívateľ výhod/KÚV na KUV
        patterns = []
        #patterns.append("podmienky na zápis členov vrcholového manažmentu podľa ust. § 4 ods. ... sú splnené")
        #patterns.append("ako spoločnosti (nepriamo) ovládanej emitentom cenných papierov")
        patterns.append("nebol (Oprávnena osoba) identifikovaná žiadna fyzická osoba, ktorá by mala viac ako 25%")
        patterns.append("sa zapisujú namiesto KÚV členovia vrcholového manažmentu")
        patterns.append("neexistuje žiadna osoba, ktorá by konala v zhode alebo spoločným postupom, ani žiadna osoba, ktorá PVS ovláda")
        patterns.append("osoba podľa [8$§] 6a ods. 2 zák.č. 297/2008")
        patterns.append("nie je žiadna fyzická osoba, ktorá v zmysle ustanovenia ... má priamy alebo nepriamy podiel")
        patterns.append("neexistuje žiadna fyzická osoba [alebo akcionár], ktorá by mala priamy alebo nepriamy podiel [alebo ich súčet] najmenej 25%")
        patterns.append("žiadna fyzická osoba nespĺňa definíciu konečného")
        patterns.append("namiesto konečných užívateľov výhod zapisuje")
        patterns.append("neidentifikovala žiadne fyzické osoby ako KUV")
        return patterns
