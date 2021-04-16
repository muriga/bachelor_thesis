import os
from urllib.request import urlopen
import urllib3
import shutil
from bs4 import BeautifulSoup
import fitz
from slearning import SupervisedClassifier
from time import sleep
import csv

PATH_DATASET = "../../Dataset/"
NEREGISTROVANY = 0
MAJITEL = 1
STATUTAR = 2


def start_findig_statutar(to_num_pvs: int):
    if not os.path.exists('vysledky'):
        os.makedirs('vysledky')
    continue_finding_statutar(1, to_num_pvs)


def continue_where_stopped():
    if not os.path.exists('vysledky'):
        os.makedirs('vysledky')
    with open('vysledky/skontrolovane.csv', 'r') as file:
        #reader = csv.reader(file, delimiter='')
        last_row = file.readlines()[-1] #TODO that not work, use rather pandas
    print(last_row)



def continue_finding_statutar(from_num_pvs: int, to_num_pvs: int):
    classifier = SupervisedClassifier(PATH_DATASET)
    classifier.train(PATH_DATASET + "majitel", PATH_DATASET + "statutar", save_model=True)
    for i in range(from_num_pvs, to_num_pvs):
        meta_data, pdf = process_detail_page(i)
        if pdf is None:
            result = NEREGISTROVANY
        elif classifier.is_owner(meta_data, pdf):
            result = MAJITEL
        else:
            result = STATUTAR
        save(meta_data, pdf, result)


def save(meta_data, pdf, result):
    if result == MAJITEL:
        state = "majitel"
    elif result == STATUTAR:
        state = "statutar"
    else:
        state = "neregistrovany"
    meta_data.append(state)
    with open('vysledky/skontrolovane.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, dialect='excel')
        writer.writerow(meta_data)
    if result == STATUTAR:
        pdf_name = 'vysledky/' + str(meta_data[9]) + ".pdf"
        pdf.save(pdf_name)


def append_if_exists(data, dict, key):
    if key in dict:
        data.append(dict[key])
    else:
        data.append("NULL")


# z div tried "form-grouo m-b-xs" vyberie slovnik atribut:hodnota
def get_attr_values_pairs(panel_body):
    rows = panel_body.find_all("div", {"class": "form-group m-b-xs"})
    d = dict()
    for row in rows:
        label = row.find("label").string.lstrip().rstrip()
        paragraph = row.find("p").string.lstrip().rstrip()
        d[label] = paragraph
    return d


# Toto prejde div PVS. Ukladá do zoznamu obchodn0 meno, ico atd. Tento zoznam potom vloží do zoznamu data
def pvs_processing(data, tag):
    d = get_attr_values_pairs(tag)
    append_if_exists(data, d, "Obchodné meno")
    append_if_exists(data, d, "IČO")
    append_if_exists(data, d, "Právna forma")
    append_if_exists(data, d, "Adresa sídla / miesto podnikania / bydliska")
    append_if_exists(data, d, "Dátum zápisu")
    append_if_exists(data, d, "Dátum výmazu")
    append_if_exists(data, d, "Číslo vložky")


# Najde obchodne meno opravnenej osoby, z data najde posledny zaznam, prida do neho toto meno
def os_processing(data, tag):
    d = get_attr_values_pairs(tag)
    if "Obchodné meno" in d:
        data.append(d["Obchodné meno"])
    else:
        data.append("NULL")


# Najde meno a priezvisko kazdeho KUV, spravi z nich jeden string s oddelovacom ' | '.
# Tento string potom pridá do posledneho záznamu z data
# Okrem toho stiahne pdf, jeho cislo tiez prida do posledneho zaznamu. Ak nie je, prida NULL
def kuv_processing(data, tag):
    table_body = tag.find("tbody")
    if table_body == None:
        data.append("NULL")
        data.append("NULL")
        return
    rows = table_body.find_all("th")
    names = ""
    for r in rows:
        name = r.string.lstrip().rstrip()
        if names != "":
            names = names + " | ";
        names = names + name
    data.append(names)


def download_pdf(data, block):
    _URL = "https://rpvs.gov.sk/"
    a = block.find("a")
    if a is None:
        data.append("NULL")
        return None
    link = a["href"]
    doc_serial_num = link[31:]
    data.append(doc_serial_num)
    url = _URL + link
    http = urllib3.PoolManager()
    doc_name = "../Dataset/all/" + doc_serial_num + ".pdf"

    request = http.request('GET', url, preload_content=False)
    bytes_pdf = request.data
    pdf = fitz.open(stream=bytes_pdf, filetype='pdf')
    return pdf


def process_detail_page(num):
    url = "https://rpvs.gov.sk/rpvs/Partner/Partner/Detail/" + str(num)
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")

    blocks = soup.find_all("div", {"class": "panel panel-default"})
    meta_data = []
    for block in blocks:
        name = block.find_all("div", {"class": "panel-heading"})
        headlines = name[0].find_all("h2")
        block = block.find_all("div", {"class": "panel-body"})[0]
        if headlines[0].string == 'Partner verejného sektora':
            pvs_processing(meta_data, block)
        elif headlines[0].string == 'Oprávnená osoba':
            os_processing(meta_data, block)
        elif headlines[0].string == 'Koneční užívatelia výhod':
            kuv_processing(meta_data, block)
            file = download_pdf(meta_data, block)
    return meta_data, file
