from urllib.request import urlopen
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import urllib3
import shutil
import csv
from random import randrange
import time


# z div tried "form-grouo m-b-xs" vyberie slovnik atribut:hodnota
def get_attr_values_pairs(panel_body):
    rows = panel_body.find_all("div", {"class":"form-group m-b-xs"})
    d = dict()
    for row in rows:
        label = row.find("label").string.lstrip().rstrip()
        paragraph = row.find("p").string.lstrip().rstrip()
        d[label] = paragraph
    return d

#Toto prejde div PVS. Ukladá do zoznamu obchodn0 meno, ico atd. Tento zoznam potom vloží do zoznamu data
def pvs_processing(data, tag):
    d = get_attr_values_pairs(tag)
    data.append(d["Obchodné meno"])
    data.append(d["IČO"])
    data.append(d["Právna forma"])
    data.append(d["Adresa sídla / miesto podnikania / bydliska"])
    data.append(d["Dátum zápisu"])
    data.append(d["Dátum výmazu"])
    data.append(d["Číslo vložky"])

#Najde obchodne meno opravnenej osoby, z data najde posledny zaznam, prida do neho toto meno
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

    link = tag.find("a")["href"]
    doc_serial_num = link[31:]
    _URL = "https://rpvs.gov.sk/"
    url = _URL + link
    c = urllib3.PoolManager()
    doc_name = "../Dataset/all/" + doc_serial_num + ".pdf"

    with c.request('GET', url, preload_content=False) as resp, open(doc_name, "wb") as out_file:
        shutil.copyfileobj(resp, out_file)

    data.append(doc_serial_num)

def process_detail_page(num):
    url = "https://rpvs.gov.sk/rpvs/Partner/Partner/Detail/" + str(num)
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")

    blocks = soup.find_all("div", {"class": "panel panel-default"})
    data = []
    for block in blocks:
        name = block.find_all("div", {"class": "panel-heading"})
        headlines = name[0].find_all("h2")
        block = block.find_all("div", {"class": "panel-body"})[0]
        if headlines[0].string == 'Partner verejného sektora':
            pvs_processing(data, block)
        elif headlines[0].string == 'Oprávnená osoba':
            os_processing(data, block)
        elif headlines[0].string == 'Koneční užívatelia výhod':
            kuv_processing(data, block)
    return data

if __name__ == "__main__":
    _LAST = 32820

    with open('../Dataset/all.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for i in range(1,12):
            writer.writerow(process_detail_page(i))
        sleeping_time = randrange(1, 2)
        time.sleep(sleeping_time)



#url = "https://rpvs.gov.sk/rpvs/Partner/Partner/VyhladavaniePartnera"
#session = HTMLSession()
#r = session.get(url)
#r.html.render()
#print(r)
#print(r.html.links)

#soup = BeautifulSoup(r.html, "html.parser")

#browser = mechanicalsoup.Browser()
#page = browser.get(url)
#soup = page.soup

#table = soup.find_all("table")

#print(table)