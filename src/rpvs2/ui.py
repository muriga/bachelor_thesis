import os
from slearning import MLPClassifierBoW
import csv
import pandas as pd
import time
import os
from webscraping import process_detail_page
from webscraping import EndRegister

PATH_DATASET = "../../Dataset/"
RESULTS_FOLDER_PATH = "../../nove_vysledky/"
CHECKED_CSV_PATH = RESULTS_FOLDER_PATH + "skontrolovane.csv"
PATH_MODEL = '../../models/model_04-17-082028.joblib'
NEREGISTROVANY = 0
MAJITEL = 1
STATUTAR = 2


class Handler:

    def __init__(self):
        self.clear = lambda: os.system('cls')
        print(f'Načítavam')
        self.classifier = MLPClassifierBoW(PATH_DATASET)
        self.classifier.train(PATH_DATASET + "majitel", PATH_DATASET + "statutar", path_pretrained=PATH_MODEL)

    def handle(self):
        self.clear()
        last_record = self.find_last_record()
        choice = -1
        while choice <= 0 or choice > 5:
            if choice == 0:
                print(f'Vyberte číslo podľa ponúkaných možností\n')
            print_options(last_record)
            user_input = input()
            choice = int(user_input) if user_input.isnumeric() else 0
            self.clear()
        try:
            self.process_choice(last_record, choice)
        except EndRegister:
            print(f'Boli skontrolované všetky záznamy z registra')
            return False
        if choice == 5:
            return False
        return True

    def process_choice(self, last_record, choice):
        if choice == 1:
            self.process_to_end(last_record)
        elif choice == 2:
            print(f'Zadajte v celých minútach približný čas na kontrolovanie: ')
            user_input = input()
            while not user_input.isnumeric():
                print(f'Minúty musia byť zadané ako číslo')
                user_input = input()
            minutes = int(user_input)
            self.process_for_time(last_record, minutes)
        elif choice == 3:
            print(f'Zadajte počet záznamov, ktoré majú byť skontrolované: ')
            user_input = input()
            while not user_input.isnumeric():
                print(f'Počet musí byť zadaný ako číslo')
                user_input = input()
            records = int(user_input)
            self.process_n_records(last_record, records)
        elif choice == 4:
            print(f'Zadajte číslo vložky záznamu, ktorý má byť skontrolovaný: ')
            user_input = input()
            while not user_input.isnumeric():
                print(f'Zadajte číslo')
                user_input = input()
            record = int(user_input)
            answer = self.process_record(record, update_logging=False)
            if answer == MAJITEL:
                print(f'PVS s č. vložky {record} má ako KUV zapísaných majiteľov.')
            elif answer == STATUTAR:
                print(f'PVS s č. vložky {record} má ako KUV zapísaných štatutárov.')
            else:
                print(f'PVS s č. vložky {record} má vymazaný záznam, preto KUV nie sú známi.')
            input("\nPre pokračovanie stlačte enter")


    def find_last_record(self):
        if not os.path.exists(RESULTS_FOLDER_PATH):
            os.makedirs(RESULTS_FOLDER_PATH)
            return 0
        with open(RESULTS_FOLDER_PATH + "logging.log", "r") as file:
            return int(file.read())

    def process_for_time(self, last_record, minutes):
        record = last_record + 1
        start_time = time.time()
        seconds = minutes * 60
        while time.time() - start_time < seconds:
            self.process_record(record)
            record += 1
        print(f'Spracovávanie ukončené, z dôvodu uplynutia stanoveného času')

    def process_to_end(self, last_record):
        record = last_record + 1
        while True:
            self.process_record(record)
            record += 1
        print(f'Boli spracované všetky záznamy z registra')

    def process_n_records(self, last_record, records):
        for i in range(last_record + 1, last_record + records + 1):
            self.process_record(i)

    def process_record(self, num, update_logging=True):
        meta_data, pdf = process_detail_page(num)
        if pdf is None:
            if update_logging:
                with open(RESULTS_FOLDER_PATH + "logging.log", "w") as file:
                    file.write(str(num))
            return 0
        elif self.classifier.is_owner(meta_data, pdf):
            result = MAJITEL
        else:
            result = STATUTAR
        save(meta_data, pdf, result)
        if update_logging:
            with open(RESULTS_FOLDER_PATH + "logging.log", "w") as file:
                file.write(str(num))
        return result


def print_options(last_record):
    print(f'Posledný skontrolovaný záznam má č. vložky {last_record}\n')  # Posledny zapis?
    print(f'Vyberte z možností ako má program prechádzať register:')
    print(f'1\tSkontrolovať od {last_record}. vložky až do posledného záznamu')
    print(f'2\tSkontrolovať od {last_record}. vložky stanovený čas')
    print(f'3\tSkontrolovať od {last_record}. vložky stanovený počet nasledujúcich záznamov')
    print(f'4\tSkontrolovať jeden konkrétny záznam')
    print(f'5\tKoniec')


def save(meta_data, pdf, result):
    if result == MAJITEL:
        state = "majitel"
    elif result == STATUTAR:
        state = "statutar"
    else:
        return
    meta_data['typ'] = state
    with open(CHECKED_CSV_PATH, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, dialect='excel')
        writer.writerow(list(meta_data.values()))
    if result == STATUTAR:
        pdf_name = RESULTS_FOLDER_PATH + '/' + str(meta_data['pdf']) + ".pdf"
        pdf.save(pdf_name)
