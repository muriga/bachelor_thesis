from template import Classifier
import stanza
import pandas as pd
import re
from ocr import convert_to_text
from ocr import get_text
from utilities import substitute


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

    def get_sentences(self, pdf_name, using_stanza=False):
        if not pdf_name.endswith(".pdf"):
            pdf_name = pdf_name + ".pdf"
        text = get_text(self.path_to_dataset + 'test_all/' + pdf_name)
        text = self.replace_meta(text, pdf_name)
        text = self.preprocessing(text)
        if using_stanza:
            text = self.lemmatize(text)
        patterns = ["([^.]| z \\.| [0-9$§]+ . | p.sm \\. | ods \\.)*pvs([^.]| z \\.|[0-9$§]+ \\. )*kuv([^.]| "
                    "z \\.| [0-9$§]+ . | p.sm \\. | ods \\.)*",
                    "([^.]| z \\.| [0-9$§]+ . | p.sm \\. | ods \\.)*kuv([^.]| z \\.|[0-9$§]+ \\. )*pvs([^.]| "
                    "z \\.| [0-9$§]+ . | p.sm \\. | ods \\.)*"]
        sentences = self.do_get_sentences(patterns, text)
        return sentences

    def do_get_sentences(self, patterns, text):
        sentences = []
        for pattern in patterns:
            found = re.finditer(pattern, text, flags=re.UNICODE)
            sentences = self.append(sentences, found)
        return sentences

    def append(self, sentences, found):
        for i in found:
            sentences.append(i.group())
        return sentences

    # def play(self, pdf_name, document_class):
    #     if not pdf_name.endswith(".pdf"):
    #         pdf_name = pdf_name + ".pdf"
    #     text = get_text(self.path_to_dataset + 'test2/' + pdf_name)
    #     text = self.replace_meta(text, pdf_name)
    #     text = self.preprocessing(text)
    #     tokenized_text = self.tokenize(text)
    #     patterns = ["([^.]| z \\.| [0-9$§]+ . | p.sm \\. | ods \\.)*pvs([^.]| z \\.|[0-9$§]+ \\. )*kuv([^.]| "
    #                 "z \\.| [0-9$§]+ . | p.sm \\. | ods \\.)*",
    #                 "([^.]| z \\.| [0-9$§]+ . | p.sm \\. | ods \\.)*kuv([^.]| z \\.|[0-9$§]+ \\. )*pvs([^.]| "
    #                 "z \\.| [0-9$§]+ . | p.sm \\. | ods \\.)*"]
    #     sentences, tokenized_sentences = self.do(patterns, text, tokenized_text)
    #     while len(sentences) > len(tokenized_sentences):
    #         tokenized_sentences.append(" ")
    #     while len(sentences) < len(tokenized_sentences):
    #         sentences.append(" ")
    #     label = [document_class for _ in sentences]
    #     return sentences, tokenized_sentences, label
    #
    # def do(self, patterns, text, tokenized_text):
    #     sentences = []
    #     tokenized_sentences = []
    #     for pattern in patterns:
    #         found = re.finditer(pattern, text, flags=re.UNICODE)
    #         found_tokenized = re.finditer(pattern, tokenized_text, flags=re.UNICODE)
    #         sentences, tokenized_sentences = self.append(sentences, tokenized_sentences, found, found_tokenized)
    #     return sentences, tokenized_sentences
    #
    # def append(self, sentences, tokenized_sentences, found, found_tokenized):
    #     for i in found:
    #         sentences.append(i.group())
    #     for i in found_tokenized:
    #         tokenized_sentences.append(i.group())
    #     return sentences, tokenized_sentences

    def is_owner_testing(self, pdf_name: str, fact_is_owner) -> bool:
        owner = True
        if not pdf_name.endswith(".pdf"):
            pdf_name = pdf_name + ".pdf"
        text = get_text(self.path_to_dataset + 'test_all/' + pdf_name)
        if text is None:
            print(f'Cannot find {pdf_name}')
            return True
        text = self.replace_meta(text, pdf_name)
        text = self.preprocessing(text)
        text = self.lemmatize(text)
        for i in range(len(self.patterns)):
            if re.search(self.patterns[i], text) is not None:
                # print(f'{pdf_name} recognized by pattern {i}')
                if fact_is_owner:
                    self.confused[i] += 1
                else:
                    self.helped[i] += 1
                owner = False
        return owner

    def pattern_statistics(self):
        POCET_STATUTAROV = 24
        for i in range(len(self.patterns)):
            if self.helped[i] + self.confused[i] == 0:
                precision_statutar = 1
            else:
                precision_statutar = self.helped[i] / (self.helped[i] + self.confused[i])
            recall_statutar = self.helped[i] / POCET_STATUTAROV
            f1_statutar = (2 * precision_statutar * recall_statutar) / (precision_statutar + recall_statutar)
            print(f'{i}. pattern: {self.helped[i]}\tzle: {self.confused[i]}')
            print(f'precision: {precision_statutar}, recall: {recall_statutar}, f1-score: {f1_statutar}')

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
            text = substitute(_kuv, 'KUV', text)
        text = substitute(_PVS, 'PVS', text)
        text = substitute(_OS, 'OS', text)
        text = substitute(_ADDR, 'ADDR', text)
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

    def lemmatize(self, text):
        lemmatized = ""
        doc = self.nlp(text)
        # for i, sentence in enumerate(doc.sentences):
        for sentence in doc.sentences:
            for token in sentence.words:
                # only if is not in stop words?
                if token.lemma not in self.stop_words:

                    if token.text[:2] == "ne" and token.lemma[:2] != "ne":
                        lemmatized += "ne" + token.lemma + " "
                    else:
                        lemmatized += token.lemma + " "
        return lemmatized

    def preprocessing(self, text):
        patterns = []
        vm_kuv = "((jeho|jej) *)?(.len([^ ]*)? *)?((jeho|jej) *)?(vrcholov[^ ]* " \
                 "*mana.ment[^ ]*|(predstav[^ ]* *)?.tatut.r[^ ]* *org.n[^ ]*)+"
        pvs = "partner([^ ]*)?( *verejn.([^ ]*)? *sektora)?"
        kuv = "kone.n[^ ]* *u..vate[^ ]* *v.hod[^ ]*"
        nofo = ".iadn[^ ]* *fyzick. *osob."
        fo = "fyzick. *osob."
        os = ".pr.vn[^ ]* *.sob[^ ]*"
        zakx = "[8$§](\.| |\n)*6a(\.| |\n)*(od)?([^ \n])*(\.| |\n)*([1-9I]?(\.| |\n)*)((p.s([^ \n])*" \
               "(\.| |\n)*([a-z)])*)?( |\n)*)?[Zz].ko([^ \n])*(\.| |\n)*(.\.(\.| |\n)*297\/2008(\.| |\n)*" \
               ".(\.| |\n)*.(\.| |\n)*)?(o( |\n))?(ochrane( |\n)*)?(pred( |\n)*)?(leg([^ \n])*( |\n)*)?" \
               "(pr.j([^ \n])*( |\n)*)?(z( |\n)*)?(tr([^ \n])*( |\n)*)?(.inn([^ \n])*(\n| ))?(a )?(o )?" \
               "(ochrane (pred )?fin([^ \n])* ter([^ \n])* . . zmene a do([^ \n])* nie([^ \n])*( |\n)*" \
               "z.k([^ \n])*( |\n))?((v )?znen([^ \n])*( |\n)nesk([^ \n])*( |\n)pred([^ \n])*)?"
        pvs2 = "p[vy][s5]"
        text = " ".join(text.split())
        text = text.lower()
        text = substitute(vm_kuv, "vmkuv", text)
        text = substitute(pvs, "pvs", text)
        text = substitute(pvs2, "pvs", text)
        text = substitute(kuv, "kuv", text)
        text = substitute(nofo, "nofo", text)
        text = substitute(fo, "fo", text)
        text = substitute(os, "os", text)
        text = substitute("sp..a", "spĺňa", text)
        text = substitute("kúv", "kuv", text)
        # text = substitute(zakx, "zak1", text)
        return text

    def get_patterns(self):
        patterns = []
        patterns.append(re.compile("ovl.d[^ ]* emitent[^ ]*"))  # 0
        patterns.append(re.compile("6a ods \. 2"))  # 1
        patterns.append(re.compile(
            r'(nofo nespĺňa|(nespĺňa )?nofo) (pvs )?(nespĺňa )?(defin.ci[^ ]*|krit.ri[^ ]*)?([^ ]* | ){0,5}kuv'))  # 2
        patterns.append(re.compile(r'namiesto kuv'))  # 3
        patterns.append(re.compile(r'nofo (pvs )*nespĺňa (krit.r(^ )*|podmienk(^ )*)'))  # 4
        patterns.append(re.compile(r'pov[^ ]* vmkuv'))  # 5
        return patterns

    def get_setting_patterns(self):
        patterns = []
        patterns.append("podmienky na zápis členov vrcholového manažmentu podľa ust. § 4 ods. ... sú splnené")
        patterns.append("ako spoločnosti nepriamo ovládanej emitentom cenných papierov")
        patterns.append("nebol os identifikovaná žiadna fyzická osoba, ktorá by mala viac ako 25%")
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
