from template import Classifier
import stanza
import pandas as pd
import re
from ocr import convert_to_text
from ocr import get_text


class PatternExtract(Classifier):
    """Class is providing evaluation if given pdf file is 'statutar'. For this purpose it use patterns"""

    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset
        self.stop_words = super().get_stop_words()
        self.nlp = stanza.Pipeline('sk', verbose=False, processors="tokenize,lemma")
        self.meta_info = pd.read_csv(path_to_dataset + "all.csv", encoding="utf-8")
        self.patterns = self.get_patterns()
        self.helped = [0 for x in self.patterns]

    def is_owner(self, pdf_name: str) -> bool:
        owner = True
        if not pdf_name.endswith(".pdf"):
            pdf_name = pdf_name + ".pdf"
        text = get_text(self.path_to_dataset + 'test2/' + pdf_name)
        if text is None:
            print(f'Cannot find {pdf_name}')
            return True
        text = self.replace_meta(text, pdf_name)
        text = self.tokenize(text)
        for i in range(len(self.patterns)):
            if re.search(self.patterns[i], text) is not None:
                # print(f'{pdf_name} recognized by pattern {i}')
                self.helped[i] += 1
                owner = False
        return owner

    def pattern_statistics(self):
        for i in range(len(self.patterns)):
            print(f'{i}. pattern: {self.helped[i]}')

    # Zákona o ochrane pred legalizáciou príjmov z trestnej činnosti a o ochrane pred |
    # inancovaním terorizmu a o zmene a doplnení niektorých zákonov -> zakx
    def replace_meta(self, text, pdf_name):
        pdf_name = int(re.findall("[0-9]+", pdf_name)[0])
        meta_info = self.meta_info.loc[self.meta_info['PDF'] == pdf_name]
        if meta_info.empty:
            # print(f'Nemam {pdf_name}')
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
        """This was used in the process of finding regex patterns"""
        tokenized_patterns = []
        for p in self.patterns:
            pattern = ""
            print(type(p))
            doc = self.nlp(p)
            # for i, sentence in enumerate(doc.sentences):
            for sentence in doc.sentences:
                for token in sentence.words:
                    # only if is not in stop words?
                    if token.lemma not in self.stop_words:
                        if token.text[:2] == "ne" and token.lemma[:2] != "ne":
                            pattern += "ne" + token.lemma + " "
                        else:
                            pattern += token.lemma + " "
            tokenized_patterns.append(pattern)
        for i in range(len(tokenized_patterns)):
            print(f'{i}. pattern: {tokenized_patterns[i]}')

    def tokenize(self, text):
        tokenized = ""
        doc = self.nlp(text)
        # for i, sentence in enumerate(doc.sentences):
        for sentence in doc.sentences:
            for token in sentence.words:
                # only if is not in stop words?
                if token.lemma not in self.stop_words:
                    if token.text[:2] == "ne" and token.lemma[:2] != "ne":
                        tokenized += "ne" + token.lemma + " "
                    else:
                        tokenized += token.lemma + " "
        return tokenized

    def get_patterns(self):
        patterns = []
        patterns.append(re.compile(r'podmienka zápis členov vrcholového manažment '
                                   r'ust \. [$8§] 4 ods[a-zA-Z0-9.,§$ ]{0,80} byť splnené'))  # 0.
        patterns.append(re.compile("spoločnosť (výlučne )?(nepriamo )?ovládanej emitentom cenných papier"))  # 1
        patterns.append(re.compile("nebyť (oprávnenou osobou )?(os )?identifikovaná žiaden fyzický osoba( ,)? mať 25%"))  # 2
        patterns.append(re.compile("zapisujú namiesto k[uú]v člen(ov)? vrcholového manažment"))  # 3
        patterns.append(re.compile("neexistovať žiaden osoba( ,)? konať zhode spoločným postupom( ,)? žiaden osoba( ,"
                                   ")? pvs ovláda"))  # 4
        patterns.append(re.compile("osoba [8$§]? 6a ods \. 2 zák \. č \. 297\/2008")) # 5
        patterns.append(re.compile(r'nie byť žiaden fyzický osoba( ,)? zmysel ustanovenie '
                                   r'zakx mať priamy nepriamy podiel')) # 6
        patterns.append(re.compile(r'neexistovať žiaden fyzický osoba( alebo akcionár)?'
                                   r'( ,)? mať priamy nepriamy podiel (alebo súčet )?(najmenej )?25%')) # 7
        patterns.append(re.compile(r'žiaden fyzický osoba nespĺňa (definícium|kritéria)?'))# (konečného|KUV)')) # 8
        patterns.append(re.compile(r'namiesto konečných užívateľ výhod zapisuje'))  # This should be better preprocessed look at #3
        patterns.append(re.compile(r'neidentifikoval(a)? žiadny fyzické osoba kuv')) # 10
        patterns.append(re.compile(r'z[aá]pis [cč]len vrcholov[eé]ho mana[zž]ment(u)?([^A-W ]*)(s[úu]|byť) splnen[eé]')) # 11
        patterns.append(re.compile(r'[čc]len(ovia)? vrcholového')) # 12
        return patterns

    def get_setting_patterns(self):
        # chcelo by to predtym zmenit veci typu konečný užívateľ výhod/KÚV na KUV
        # namiesto KUV niečo
        # zapisujú namiesto konečných užívateľov výhod štatutárny orgán #9,3
        # sú splnené podmienky na zápis členov štatutárneho orgánu do registra namiesto konečného užívateľa výhod #1,3,11
        # namiesto konečného užívateľa výhod zapisujú členovia jeho štatutárneho orgánu #9,3
        # budú v súlade s § 4 odsek 5 zákona č. 315/2016 o registri partnerov verejného sektora
        # zapísaný členovia predstavenstva štatutárneho orgánu PVS    #zapisany clenovia predstavenstva?
        patterns = []
        patterns.append("podmienky na zápis členov vrcholového manažmentu podľa ust. § 4 ods. ... sú splnené")
        patterns.append("ako spoločnosti (nepriamo) ovládanej emitentom cenných papierov")
        patterns.append("nebol (Oprávnena osoba) identifikovaná žiadna fyzická osoba, ktorá by mala viac ako 25%")
        patterns.append("sa zapisujú namiesto KÚV členovia vrcholového manažmentu") # štatutárny orgán
        patterns.append(
            "neexistuje žiadna osoba, ktorá by konala v zhode alebo spoločným postupom, ani žiadna osoba, ktorá PVS ovláda")
        patterns.append("osoba podľa [8$§] 6a ods. 2 zák.č. 297/2008")
        patterns.append("nie je žiadna fyzická osoba, ktorá v zmysle ustanovenia ... má priamy alebo nepriamy podiel")
        patterns.append(
            "neexistuje žiadna fyzická osoba [alebo akcionár], ktorá by mala priamy alebo nepriamy podiel [alebo ich súčet] najmenej 25%")
        patterns.append("žiadna fyzická osoba nespĺňa definíciu konečného")
        patterns.append("namiesto konečných užívateľov výhod zapisuje")
        patterns.append("neidentifikovala žiadne fyzické osoby ako kuv")
        patterns.append(r'zápis členov vrcholového manažmentu sú splnené')
        patterns.append(r'členovia vrcholového')
        return patterns

    def test(self):
        strings = []
        strings.append("podmienka zápis členov vrcholového manažment ust . § 4 ods . . . . byť splnené")
        strings.append("spoločnosť nepriamo ovládanej emitentom cenných papier")
        strings.append("nebyť os identifikovaná žiaden fyzický osoba , mať 25%")
        strings.append("zapisujú namiesto kúv člen vrcholového manažment")
        strings.append("neexistovať žiaden osoba , konať zhode spoločným postupom , žiaden osoba , pvs ovláda")
        strings.append("osoba § 6a ods . 2 zák . č . 297/2008")
        strings.append("nie byť žiaden fyzický osoba , zmysel ustanovenie zakx mať priamy nepriamy podiel")
        strings.append("neexistovať žiaden fyzický osoba alebo akcionár , mať "
                       "priamy nepriamy podiel alebo súčet najmenej 25%")
        strings.append("žiaden fyzický osoba nespĺňa definícium konečného")
        strings.append("namiesto konečných užívateľ výhod zapisuje")
        strings.append("neidentifikovala žiadny fyzické osoba kuv")
        for i in range(len(self.patterns)):
            if re.search(self.patterns[i], strings[i]) is None:
                print(i)
                print(self.patterns[i])
                print(strings[i])
