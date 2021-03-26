"""
Get text from pdf

This module provide methods which convert both searchable and scanned pdfs to string.
It allows user to give just one file, path to file or folder.
Script can save output string to file with same name as pdf have
or return. If path to folder was given, script returns dir of strings.

Dependencies:   fitz
                pytesseract
                pillow
                cv2

Module provides following functions:
    * convert_to_text - converts one pdf file to string
    * iterate_folder_convert_to_text - converts every pdf from folder to separate string
    * get_text - get text of one pdf file which was earlier recognized and saved
    * iterate_folder_get_text - returns dictionary of separate strings from txt files of given folder
"""
from typing import Union
from PIL import Image
from PIL.Image import FLIP_TOP_BOTTOM
from io import BytesIO
from re import search
from numpy import asarray
from pytesseract import pytesseract, TesseractError
import fitz
import os


def convert_to_text(file: Union[str, fitz.Document], save: bool = False):
    """Gets file or path to file, convert its to string and if save is set to True, save string
    to txt file with same name as pdf have. Method also returns that string.


    :param file: The file opened by fitz or path to the file
    :type file: Union[str, fitz.Document]
    :param save: Determine if string should be saved. Default is false.
    :type save: bool
    :returns: Recognized text
    :rtype: str

    """
    text = ""
    if type(file) == str:
        file = fitz.open(file)
    images = get_images(file)
    if len(images) > 0:
        text = images_to_string(images)
    text += searchable_pdf_parts_to_string(file)
    if save:
        txt_file_name = file.name.removesuffix("pdf") + "txt"
        with open(txt_file_name, "w", encoding="utf-8") as file:
            file.write(text)
    return text


def iterate_folder_convert_to_text(folder: str, save: bool = False, contains_txt=False):
    """Gets file to folder, then iterate thought every pdf file in this folder, convert
        it to string. Then save text to given dict or when no dict was given, to txt files
        in same folder.

    :param folder: Path to folder with pdf files which should be converted to text.
    :type folder: str
    :param save: Determine if strings should be saved. Default is false.
    :type save: bool
    :param contains_txt: If function should make txt files, this determine if it will check if txt for certain pdf exists.
    If folder contains them, function ignore that pdf
    :type contains_txt: bool
    :returns: A dictionary where key is name of file and value recognized text
    :rtype: dict
    """
    files = dict()
    directory = os.fsencode(folder)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        txt_file = os.fsencode(filename.removesuffix(".pdf") + ".txt")
        if filename.endswith(".pdf") and (not contains_txt or txt_file not in os.listdir(directory)):
            file_path = folder + "/" + filename
            text = convert_to_text(file_path)
            files[filename.title().removesuffix(".Pdf")] = text
            if save:
                txt_file_name = file_path.removesuffix("pdf") + "txt"
                with open(txt_file_name, "w", encoding="utf-8") as file:
                    file.write(text)
    return files


def get_text(file_path: str) -> str:
    """When path to pdf of txt is given, function return string which was earlier recognized. If it can't find txt,
        returns None

    :param file_path: Path to pdf or txt file
    :type file_path: str
    :returns: Text which was earlier recognized
    :rtype: str
    """
    if file_path.endswith(".pdf"):
        file_path = file_path.removesuffix(".pdf") + ".txt"
    elif file_path.endswith(".txt"):
        pass
    else:
        file_path = file_path + ".txt"
    try:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = None
    return text


def iterate_folder_get_text(folder_path: str) -> str:
    """When path to folder is given, function return dict of strings which was earlier recognized. Function considers
        only txt files.

    :param folder_path: Path to folder
    :type folder_path: str
    :returns: Dictionary where keys are names of txt/pdf files and values its text
    :rtype: dict
    """
    files = dict()
    directory = os.fsencode(folder_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            file_path = folder_path + "/" + filename
            text = get_text(file_path)
            files[filename.title().removesuffix(".Txt")] = text
    return files


def get_images(pdf_file):
    """Gets opened pdf file, return list of images from that pdf"""
    # Source: https://www.thepythoncode.com/article/extract-pdf-images-in-python
    pages = len(pdf_file)
    images = []
    for i in range(pages):
        page = pdf_file[i]
        for image_index, img in enumerate(page.getImageList(), start=1):
            # TODO may be better to iterate through all img
            xref = img[0]
            base_image = pdf_file.extractImage(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))
            images.append(image)
            # This may be useful while debugging
            image_ext = base_image["ext"]
            image.save(open(f"{pdf_file.name.removesuffix('.pdf')}_{i}_{image_index}.{image_ext}", "wb"))
    return images


def contain_too_many_uppercase(text):
    i = 25
    uppercase = 0
    lowercase = 0
    other = 0
    while i < 125 and i < len(text):
        if text[i].isupper():
            uppercase += 1
        elif text[i].islower():
            lowercase += 1
        else:
            other += 1
        i += 1
    if lowercase+uppercase > 0 and uppercase > lowercase:
        return True
    return False


def images_to_string(images):
    """Using pytesseract, method recognize text from list of images and return one string"""
    text = ""
    for i in range(len(images)):
        # TODO: contain_too_many_uppercase is maybe not best approach
        # TODO possible fixes:  1. create dictionary of common words
        # TODO                  2. tesseract recognition of language
        # TODO                  3. if one of image is damaged, all pdf are damaged
        image = images[i]
        page_text = pytesseract.image_to_string(image, lang="slk")
        # skus ci je strana ok ak nie, otoc ju, skus ci je nova ok. ak nie, rotuj left -> right skus znova
        if contain_too_many_uppercase(page_text):
            flipped_image = image.transpose(FLIP_TOP_BOTTOM)
            flipped_page_text = pytesseract.image_to_string(flipped_image, lang="slk")
            if contain_too_many_uppercase(flipped_page_text):
                balanced_image = balance_skew(image)
                balanced_page_text = pytesseract.image_to_string(balanced_image, lang="slk")
                if not contain_too_many_uppercase(balanced_page_text):
                    page_text = balanced_page_text
            else:
                page_text = flipped_page_text
        text += page_text
    return text


def searchable_pdf_parts_to_string(pdf_file):
    """Method returns all text stored in pdf_file as text in one string"""
    text = ""
    for page in pdf_file.pages():
        text += page.get_textpage().extractText()
    return text


def balance_skew(image):
    """Using pytesseract method recognize skew of text in image and returns balanced image"""
    cv_image = asarray(image)
    try:
        information = pytesseract.image_to_osd(cv_image)
        rotation = search('(?<=Rotate: )\d+', information).group(0)
        return image.rotate(-int(rotation))
    except TesseractError:
        print("Havent rotated - tesseract error")
    return image
