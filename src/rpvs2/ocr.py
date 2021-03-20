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

Module provides following methods:
    * convert_to_text - converts one pdf file to string
    * iterate_folder_convert_to_text - converts every pdf from folder to separate string
"""
from typing import Union
from PIL import Image
from PIL.Image import FLIP_LEFT_RIGHT
from io import BytesIO
from re import search
from numpy import asarray
from pytesseract import pytesseract
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


def iterate_folder_convert_to_text(folder: str, save: bool = False, contains_txt = False):
    """Gets file to folder, then iterate thought every pdf file in this folder, convert
        it to string. Then save text to given dict or when no dict was given, to txt files
        in same folder.

    :param folder: Path to folder with pdf files which should be converted to text.
    :type folder: str
    :param save: Determine if strings should be saved. Default is false.
    :type save: bool
    :param contains_txt: If method should make txt files, this determine if it will check if txt for certain pdf exists.
    Method doesn't replace such txt if param is true.
    :type contains_txt: bool
    :returns: A dictionary where key is name of file and value recognized text
    :rtype: dict
    """
    files = dict()
    directory = os.fsencode(folder)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pdf"):
            file_path = folder + "/" + filename
            text = convert_to_text(file_path)
            files[filename.title().removesuffix(".Pdf")] = text
            if save:
                txt_file = os.fsencode(filename.removesuffix(".pdf") + ".txt")
                if not contains_txt or txt_file not in os.listdir(directory):
                    txt_file_name = file_path.removesuffix("pdf") + "txt"
                    with open(txt_file_name, "w", encoding="utf-8") as file:
                        file.write(text)


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
            # image_ext = base_image["ext"]
            # image.save(open(f"image{pdf_file.removesuffix('.pdf')}_{image_index}.{image_ext}", "wb"))
    return images


def images_to_string(images):
    """Using pytesseract, method recognize text from list of images and return one string"""
    text = ""
    for i in range(len(images)):
        # TODO: Q: is balancing skew necessary? What about trying balancing a left to right
        # TODO: transpose only if recognition is bad?
        # TODO: When document is flipped left to right... how to recognize that?
        rotated_image = balance_skew(images[i])
        page_text = pytesseract.image_to_string(rotated_image, lang="slk")
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
    information = pytesseract.image_to_osd(cv_image)
    rotation = search('(?<=Rotate: )\d+', information).group(0)
    return image.rotate(-int(rotation))

def is_in_folder(filename,folder):
    print(filename)
    print(folder)
    print(filename in folder)
    return True