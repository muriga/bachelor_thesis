from template import Classifier
import stanza
import pandas as pd
import re
from ocr import convert_to_text
from typing import Union
import fitz
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

    def is_owner(self, meta_data: list, pdf: Union[str, fitz.Document]):
        if not pdf.endswith(".pdf"):
            pdf = pdf + ".pdf"
        text = get_text(self.path_to_dataset + 'test_all/' + pdf)
        if text is None:
            return True
        text = self.replace_meta(text, pdf)
        text = self.preprocessing(text)
        text = self.lemmatize(text)
        for i in range(len(self.patterns)):
            if re.search(self.patterns[i], text) is not None:
                return False
        return True

    def is_owner_testing(self, pdf_name: str, fact_is_owner) -> bool:
        owner = True
        if not pdf_name.endswith(".pdf"):
            pdf_name = pdf_name + ".pdf"
        text = get_text(self.path_to_dataset + 'test_all/' + pdf_name)
        if text is None:
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
        POCET_STATUTAROV = 50
        for i in range(len(self.patterns)):
            if self.helped[i] + self.confused[i] == 0:
                precision_statutar = 1
            else:
                precision_statutar = self.helped[i] / (self.helped[i] + self.confused[i])
            recall_statutar = self.helped[i] / POCET_STATUTAROV
            f1_statutar = (2 * precision_statutar * recall_statutar) / (precision_statutar + recall_statutar)
            print(f'{i}. pattern: {self.helped[i]}\tzle: {self.confused[i]}')
            print(f'precision: {precision_statutar}, recall: {recall_statutar}, f1-score: {f1_statutar}')

    def replace_meta(self, text, pdf_name):
        pdf_name = int(re.findall("[0-9]+", pdf_name)[0])
        meta_info = self.meta_info.loc[self.meta_info['PDF'] == pdf_name]
        if meta_info.empty:
            raise Exception(f'Unable to use metainformations - I can\'t find PVS with name of pdf: \'{pdf_name}\'')
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

    def lemmatize(self, text):
        lemmatized = ""
        doc = self.nlp(text)
        for sentence in doc.sentences:
            for token in sentence.words:
                if token.lemma not in self.stop_words:
                    if token.text[:2] == "ne" and token.lemma[:2] != "ne":
                        lemmatized += "ne" + token.lemma + " "
                    else:
                        lemmatized += token.lemma + " "
        return lemmatized

    def preprocessing(self, text):
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
        patterns = [re.compile("ovl.d[^ ]* emitent[^ ]*"),
                    re.compile("6a ods \. 2"),
                    re.compile(
                        r'(nofo nespĺňa|(nespĺňa )?nofo) (pvs )?(nespĺňa )?(defin.ci[^ ]*|krit.ri[^ ]*)?([^ ]* | ){0,5}kuv'),
                    re.compile(r'namiesto kuv'), re.compile(r'nofo (pvs )*nespĺňa (krit.r(^ )*|podmienk(^ )*)'),
                    re.compile(r'pov[^ ]* vmkuv')]
        return patterns
