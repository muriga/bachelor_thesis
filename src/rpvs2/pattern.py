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
        text = get_text(self.path_to_dataset + 'test3/' + pdf_name)
        if text is None:
            print(f'Cannot find {pdf_name}')
            return True
        text = self.replace_meta(text, pdf_name)
        text = self.preprocessing(text)
        text = self.tokenize(text)
        for i in range(len(self.patterns)):
            if re.search(self.patterns[i], text) is not None:
                print(f'{pdf_name} recognized by pattern {i}')
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
        nofo = "(.iadn[^ \n]*)?( |\n)*fyzick.( |\n)*osoba"
        fo = "fyzick.( |\n)*osob."
        os = ".pr.vn[^ \n]*( |\n)*.sob[^ \n]*"
        zakx = "[8$§](\.| |\n)*6a(\.| |\n)*od([^ \n])*(\.| |\n)*([1-9I]?(\.| |\n)*)((p.s([^ \n])*" \
               "(\.| |\n)*([a-z)])*)?( |\n)*)?[Zz].ko([^ \n])*(\.| |\n)*(.\.(\.| |\n)*297\/2008(\.| |\n)*" \
               ".(\.| |\n)*.(\.| |\n)*)?(o( |\n))?(ochrane( |\n)*)?(pred( |\n)*)?(leg([^ \n])*( |\n)*)?" \
               "(pr.j([^ \n])*( |\n)*)?(z( |\n)*)?(tr([^ \n])*( |\n)*)?(.inn([^ \n])*(\n| ))?(a )?(o )?" \
               "(ochrane (pred )?fin([^ \n])* ter([^ \n])* . . zmene a do([^ \n])* nie([^ \n])*( |\n)*" \
               "z.k([^ \n])*( |\n))?((v )?znen([^ \n])*( |\n)nesk([^ \n])*( |\n)pred([^ \n])*)?"
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
        #patterns.append(re.compile(r'podmienka zápis členov vrcholového manažment '
        #                           r'ust \. [$8§] 4 ods[a-zA-Z0-9.,§$ ]{0,80} byť splnené'))  # 0.
        #patterns.append(re.compile("spoločnosť (výlučne )?(nepriamo )?ovládanej emitentom cenných papier"))  # 1
        #patterns.append(re.compile("nebyť (oprávnenou osobou )?(os )?identifikovaná žiaden fyzický osoba( ,)? mať 25%"))  # 2
        #patterns.append(re.compile("zapisujú namiesto k[uú]v člen(ov)? vrcholového manažment"))  # 3
        #patterns.append(re.compile("neexistovať žiaden osoba( ,)? konať zhode spoločným postupom( ,)? žiaden osoba( ,"
        #                           ")? pvs ovláda"))  # 4
        #patterns.append(re.compile("osoba [8$§]? 6a ods \. 2 zák \. č \. 297\/2008")) # 5
        #patterns.append(re.compile(r'nie byť žiaden fyzický osoba( ,)? zmysel ustanovenie '
        #                           r'zakx mať priamy nepriamy podiel')) # 6
        #patterns.append(re.compile(r'neexistovať žiaden fyzický osoba( alebo akcionár)?'
        #                           r'( ,)? mať priamy nepriamy podiel (alebo súčet )?(najmenej )?25%')) # 7
        #patterns.append(re.compile(r'žiaden fyzický osoba nespĺňa (definícium|kritéria)?'))# (konečného|KUV)')) # 8
        #patterns.append(re.compile(r'namiesto konečných užívateľ výhod zapisuje'))  # This should be better preprocessed look at #3
        #patterns.append(re.compile(r'neidentifikoval(a)? žiadny fyzické osoba kuv')) # 10
        #patterns.append(re.compile(r'z[aá]pis [cč]len vrcholov[eé]ho mana[zž]ment(u)?([^A-W ]*)(s[úu]|byť) splnen[eé]')) # 11
        #patterns.append(re.compile(r'[čc]len(ovia)? vrcholového')) # 0
        patterns.append(re.compile("(spoločnosť|pvs) (výlučne )?(nepriamo )?ovládanej emitentom"))  # 0
        patterns.append(re.compile("zapisuj[^ \n] namiesto kuv"))  # 1
        patterns.append(re.compile("osoba [8$§]? 6a ods \. 2 zák \. č \. 297\/2008")) #2
        patterns.append(re.compile(r'(nofo nesp..a|nesp..a nofo)( |\n)((u) pvs )?(defin.ciu[^ \n]*|krit.ri[^ \n]*|([^ \n]*( |\n)){0,3})kuv'))#3
        patterns.append(re.compile(r'namiesto kuv (zapisuje|pova.[^ \n]*)'))  #4
        patterns.append(re.compile(r'namiesto kuv')) #5
        patterns.append(re.compile(r'6a[^ \n]*ods.[^ \n]*2[^ \n]*z.kona[^ \n]*.[^ \n]*297/2008')) #6
        patterns.append(re.compile(r'nofo( |\n)*(pvs( |\n)*)*nesp..a( |\n)*(krit.r(^ \n)*|podmienk(^ \n)*)'))#7
        patterns.append(re.compile(r'kuv( |\n)*(spolo([^ \n]*)?( |\n))?pov([^ \n]*)? vmkuv'))#8
        patterns.append(re.compile(r'pov([^ \n]*)? vmkuv'))#9
        #pvs nie byť pvs
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

    def test2(self):
        text = "žiadna fyzická osoby nespíňala kritériá pre KÚV\n"\
        "KÚV vmkuv\n"\
        "----\n"\
        "žiadna fyzická osoba nespĺňa kritériá\n"\
        "za konečných užívateľov výhod PVS v zmysle § 6a ods. 2 zákona č.297/2008 Z.z. považujú vmkuv\n"\
        "---\n"\
        "pričom žiaden z akcionárov VEOLIA ENVIRRONEMENT-VE nevlastní 25 % a viac akcií\n"\
        "neexistuje fyzická osoba, ktorá disponuje právom (nepriamym) na hospodársky prospech najmenej 25 % z podnikania Partnera verejného sektora\n"\
        "---\n"\
        "žiadna fyzická osoba v PVS nesplňa kritéria uvedené v 8 6a odsek 1 písm. a) zákona č. 297/2008 Z.z. o ochrane pred legalizáciou príjmov z trestnej činnosti a:o ochrane pred financovaním terorizmu a o zmene a doplnení niektorých zákonov\n"\
        "za konečného užívateľa výhod považujú vmkuv PVS\n"\
        "---\n"\
        "sa nepodarilo identifikovať takú fyzickú osobu, resp. fyzické osoby, ktoré by spĺňali kritériá uvedené v  §6a ods. 1, písm. a)" \
        "V nadväznosti na uvedené ustanovenie sa majú podľa § 6a ods. 2 Zákona o ochrane pred legalizáciou príjmov z trestnej činnosti za konečných uţívateľov výhod Spoločnosti povaţovať vmkuv\n"\
        "---\n"\
        "Za končeného užívateľa výhod u Partnera verejného sektora sa preto vzhľadom k ust. $ 6a ods. 2\n"\
        "zákona o ochrane pred legalizáciou príjmov z trestnej činnosti považujú vmkuv. Za konečných užívateľov výhod sa považuje vmkuv, alebo vmkuv\n"\
        "---\n"\
        "žiadna fyzická osoba nespĺňa kritériá uvedené v odseku 8, za konečných užívateľov výhod sa \n"\
        "považujú členovia jej vrcholového manažmentu\n"\
        "---\n"\
        "PVS je spoločnosť, ktorú nepriamo a výlučne majetkovo ovláda a nepriamo a výlučne riadi spoločnosť, ktorá je emitentom cenných\n"\
        "do registra partnerov verejného sektora budú v súlade s § 4 odsek 5 zákona č. 315/2016 o registri partnerov verejného sektora zapísaný členovia predstavenstva štatutárneho orgánu PVS\n"\
        "---\n"\
        "že žiadna fyzická osoba nespíňa kritériá uvedené v 8 6a odsek 1 písm. a) zákona č. 297/2008 Z. z. o ochrane pred legalizáciou príjmov z trestnej činnosti a o ochrane pred financovaním terorizmu a o zmene a doplnení niektorých zákonov v znení neskorších predpisov \n"\
        "Za konečných užívateľov výhod PVS sa preto vzhľadom k 8 6a ods. 2 zákona o ochrane pred legalizáciou príjmov z trestnej činnosti považujú členovia jej vrcholového manažmentu\n"\
        "---\n"\
        "Vzhľadom na skutočnosť, že na základe predložených dokumentov a overených informácií\n"\
        "nespíňa žiadna osoba u partnera verejného sektora kritériá pre konečného užívateľa výhod\n"\
        "v zmysle 86a ods. 1 zákona č. 297/2008 Z. z. o ochrane pred legalizáciou príjmov z trestnej\n"\
        "činnosti, považujú sa za konečných užívateľov výhod u partnera verejného sektora členovia jeho\n"\
        "vrcholového manažmentu \n"\
        "---\n"\
        "žiadna fyzická osoba nespĺňa kritériá uvedené v § 6a ods. 1 písm. a) Zákona o ochrane pred legalizáciou príjmov z TČ, \n"\
        "za konečných užívateľov výhod sa považujú členovia vrcholového manažmentu Partnera verejného sektora\n"\
        "---\n"\
        "žiadna fyzická osoba nespĺňa kritériá\n"\
        "za konečných užívateľov výhod u PVS považujú členovia jeho vrcholového manažmentu\n"\
        "---\n"\
        "žiadna fyzická osoba nespíňa podmienku stanovenú ust. $ 6a ods. | aods. 3 Zákona oochrane pred legalizáciou príjmov z trestne\n"\
        "žiadnu fyzickú osobu nemožno určiť ako konečného užívateľa výhod, nakoľko žiadna fyzická osoba nespíňa kritéria uvedené v ust. $ 6a ods. I písm. a) Zákona o ochrane pred legalizáciou príjmov z trestnej činnostij činnosti\n"\
        "Za konečných užívateľov výhod u PYS sa budú v súlade s ust. 8 6a ods. 2 Zákona o ochrane príjmov z trestnej činnosti považovať členovia jej vrcholového manažmentu\n"\
        "---\n"\
        "neexistuje a nie je (im) známa žiadna fyzická osoba: -  ktorá by mala priamy podiel alebo ich súčet najmenej 25% na hlasovacích právach v Partnerovi verejného sektora\n"\
        "neexistuje žiadna fyzická osoba, ktorá by spĺňala podmienky v zmysle § 6a ods.1 písm. a) zákona č. \n"\
        "297/2008 Z.z., a preto v súlade s §6a ods.2 tohto zákona sa za konečného užívateľa výhod považujú členovia jej vrcholového manažmentu (t.z. štatutárny orgán, člen štatutárneho orgánu, prokurista a vedúci zamestnanec v priamej riadiacej pôsobnosti štatutárneho orgánu)\n"\
        "---\n"\
        "Žiadna fyzická osoba nespíňa kritériá uvedené v odseku 1 písm. a), zákona č. 279/2008 Z. z. a\n"\
        "za konečných užívateľov výhod sa považujú členovia jej vrcholového manažmentu: za člena\n"\
        "vrcholového manažmentu sa považuje štatutárny orgán, člen štatutárneho orgánu,\n"\
        "prokurista a vedúci zamestnanec v priamej riadiacej pôsobnosti štatutárneho orgánu."
        text = self.preprocessing(text)
        print(self.tokenize(text))

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
