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
import fitz

def convert_to_text(file: Union[str, fitz.Document], destination: str = None):
    """Gets file or path to file, convert its to string and if destination is set, save
        string to destination, else to txt file with same name as pdf have.

    :param file: str or fitz.Document
        The file opened by fitz or path to the file
    :param destination: str, optional
        String where output should be saved
    """
    if destination is None:
        pass

Dictonary = dict[str, str]
def iterate_folder_convert_to_text(file: str, destination: dict = None):


def convert_scanned_pdf_to_string(filename):
    """Me"""
