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
        self.confused = [0 for x in self.patterns]

    def is_owner(self, pdf_name: str, fact_is_owner) -> bool:
        owner = True
        if not pdf_name.endswith(".pdf"):
            pdf_name = pdf_name + ".pdf"
        text = get_text(self.path_to_dataset + 'test2/' + pdf_name)
        if text is None:
            print(f'Cannot find {pdf_name}')
            return True
        text = self.replace_meta(text, pdf_name)
        text = self.preprocessing(text)
        text = self.tokenize(text)
        for i in range(len(self.patterns)):
            if re.search(self.patterns[i], text) is not None:
                #print(f'{pdf_name} recognized by pattern {i}')
                if fact_is_owner:
                    self.confused[i] += 1
                else:
                    self.helped[i] += 1
                owner = False
        return owner

    def pattern_statistics(self):
        for i in range(len(self.patterns)):
            print(f'{i}. pattern: {self.helped[i]}\tzle: {self.confused[i]}')

    # Zákona o ochrane pred legalizáciou príjmov z trestnej činnosti a o ochrane pred |
    # inancovaním terorizmu a o zmene a doplnení niektorých zákonov -> zakx
    def replace_meta(self, text, pdf_name):
        pdf_name = int(re.findall("[0-9]+", pdf_name)[0])
        meta_info = self.meta_info.loc[self.meta_info['PDF'] == pdf_name]
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
            text = self.substitute(_kuv, 'KUV', text)
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

    def preprocessing(self, text):
        patterns = []
        vm_kuv = "((jeho|jej)( |\n)*)?(.len([^ \n]*)?( |\n)*)?((jeho|jej)( |\n)*)?(vrcholov[^ \n]*" \
                 "( |\n)*mana.ment[^ \n]*|(predstav[^ \n]*( |\n)*)?.tatut.r[^ \n]*( |\n)*org.n[^ \n]*)+"
        pvs = "[Pp]artner([^ \n]*)?(( |\n)*verejn.([^ \n]*)?( |\n)*sektora)?"
        kuv = "[Kk]one.n[^ \n]*( |\n)*u..vate[^ \n]*( |\n)*v.hod[^ \n]*"
        nofo = "((.iadn[^ \n]*)( |\n)*)+fyzick.( |\n)*osob."
        fo = "fyzick.( |\n)*osob."
        os = ".pr.vn[^ \n]*( |\n)*.sob[^ \n]*"
        zakx = "[8$§](\.| |\n)*6a(\.| |\n)*(od)?([^ \n])*(\.| |\n)*([1-9I]?(\.| |\n)*)((p.s([^ \n])*" \
               "(\.| |\n)*([a-z)])*)?( |\n)*)?[Zz].ko([^ \n])*(\.| |\n)*(.\.(\.| |\n)*297\/2008(\.| |\n)*" \
               ".(\.| |\n)*.(\.| |\n)*)?(o( |\n))?(ochrane( |\n)*)?(pred( |\n)*)?(leg([^ \n])*( |\n)*)?" \
               "(pr.j([^ \n])*( |\n)*)?(z( |\n)*)?(tr([^ \n])*( |\n)*)?(.inn([^ \n])*(\n| ))?(a )?(o )?" \
               "(ochrane (pred )?fin([^ \n])* ter([^ \n])* . . zmene a do([^ \n])* nie([^ \n])*( |\n)*" \
               "z.k([^ \n])*( |\n))?((v )?znen([^ \n])*( |\n)nesk([^ \n])*( |\n)pred([^ \n])*)?"
        pvs2 = "[pP][VvY][S5]"
        text = self.substitute(vm_kuv, "vmkuv", text)
        text = self.substitute(pvs, "pvs", text)
        text = self.substitute(kuv, "kuv", text)
        text = self.substitute(nofo, "nofo", text)
        text = self.substitute(fo, "fo", text)
        text = self.substitute(os, "os", text)
        text = self.substitute("sp..a", "spĺňa", text)
        text = self.substitute("KÚV", "kuv", text)
        text = self.substitute(zakx, "zak1", text)
        return text

    def get_patterns(self):
        patterns = []
        patterns.append(re.compile("(spolo.nos.|pvs) (v.lu.ne )?(nepriamo )?ovl.d(^ \n)* emitent(^ \n)*"))  # 0
        patterns.append(re.compile("osoba [8$§]? 6a ods \. 2 z.k \. . \. 297\/2008"))  # 1
        patterns.append(re.compile(
            r'(nofo nesp..a|nesp..a nofo)( |\n)((u) pvs )?(defin.ciu[^ \n]*|krit.ri[^ \n]*|([^ \n]*( |\n)){0,3})kuv'))  # 2
        patterns.append(re.compile(r'namiesto kuv'))  # 3
        patterns.append(re.compile(r'nofo( |\n)*(pvs( |\n)*)*nesp..a( |\n)*(krit.r(^ \n)*|podmienk(^ \n)*)'))  # 4
        patterns.append(re.compile(r'pov([^ \n]*)? vmkuv'))  # 5
        return patterns

    def get_setting_patterns(self):
        patterns = []
        patterns.append("podmienky na zápis členov vrcholového manažmentu podľa ust. § 4 ods. ... sú splnené")
        patterns.append("ako spoločnosti (nepriamo) ovládanej emitentom cenných papierov")
        patterns.append("nebol (Oprávnena osoba) identifikovaná žiadna fyzická osoba, ktorá by mala viac ako 25%")
        patterns.append("sa zapisujú namiesto KÚV členovia vrcholového manažmentu")  # štatutárny orgán
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
