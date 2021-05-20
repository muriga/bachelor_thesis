from urllib.request import urlopen
import urllib3
from bs4 import BeautifulSoup
import fitz

BASE_URL = "https://rpvs.gov.sk/rpvs/Partner/Partner/Detail/"

class EndRegister(Exception):
    pass

def append_if_exists(meta_data, dict, key):
    if key in dict:
        meta_data[key] = dict[key]
    else:
        meta_data[key] = "NULL"


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
def pvs_processing(meta_data, tag):
    d = get_attr_values_pairs(tag)
    append_if_exists(meta_data, d, "Obchodné meno")
    append_if_exists(meta_data, d, "IČO")
    append_if_exists(meta_data, d, "Právna forma")
    append_if_exists(meta_data, d, "Adresa sídla / miesto podnikania / bydliska")
    append_if_exists(meta_data, d, "Dátum zápisu")
    append_if_exists(meta_data, d, "Dátum výmazu")
    append_if_exists(meta_data, d, "Číslo vložky")


# Najde obchodne meno opravnenej osoby, z data najde posledny zaznam, prida do neho toto meno
def os_processing(meta_data, tag):
    d = get_attr_values_pairs(tag)
    if d is not None:
        d['os'] = d.pop('Obchodné meno')
    append_if_exists(meta_data, d, 'os')


# Najde meno a priezvisko kazdeho KUV, spravi z nich jeden string s oddelovacom ' | '.
# Tento string potom pridá do posledneho záznamu z data
def kuv_processing(meta_data, tag):
    table_body = tag.find("tbody")
    if table_body is None:
        meta_data['KUV'] = 'NULL'
        return
    rows = table_body.find_all("th")
    names = ""
    for r in rows:
        name = r.string.lstrip().rstrip()
        if names != "":
            names = names + " | ";
        names = names + name
    meta_data['KUV'] = names


def download_pdf(meta_data, block):
    _URL = "https://rpvs.gov.sk/"
    a = block.find("a")
    if a is None:
        meta_data['pdf'] = 'NULL'
        return None
    link = a["href"]
    pdf_name = link[31:]

    meta_data['pdf'] = pdf_name
    url = _URL + link
    http = urllib3.PoolManager()

    request = http.request('GET', url, preload_content=False)
    bytes_pdf = request.data
    pdf = fitz.open(stream=bytes_pdf, filetype='pdf')
    return pdf


def process_detail_page(num):
    url = BASE_URL + str(num)
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")

    blocks = soup.find_all("div", {"class": "panel panel-default"})
    meta_data = {}
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
    if meta_data['Dátum zápisu'] == '01.01.0001':
        raise EndRegister
    return meta_data, file
