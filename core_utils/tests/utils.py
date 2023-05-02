"""
Utils for core_utils tests
"""
import shutil

from config.test_params import TEST_FILES_FOLDER, TEST_PATH
from core_utils.constants import ASSETS_PATH


def universal_setup() -> None:
    """
    Creation of required assets for the core_utils unit tests
    """
    TEST_PATH.mkdir(exist_ok=True)
    shutil.copyfile(TEST_FILES_FOLDER / "1_raw.txt",
                    TEST_PATH / "1_raw.txt")
    shutil.copyfile(TEST_FILES_FOLDER / "1_meta.json",
                    TEST_PATH / "1_meta.json")


def copy_student_data() -> None:
    """
    Copy student data to safe place for tests needs
    """
    TEST_PATH.mkdir(exist_ok=True)
    for file in ASSETS_PATH.iterdir():
        shutil.copyfile(ASSETS_PATH / file.name, TEST_PATH / file.name)


# pylint: disable=too-few-public-methods
class TestInputs:
    """
    Input data for article tests
    """
    text = "Мама красиво мыла раму. Мама красиво мыла раму... " \
           "Мама красиво мыла раму! Мама красиво мыла раму!!! " \
           "Мама красиво мыла раму? Мама красиво мыла раму?! " \
           "Мама мыла раму... красиво. Мама сказала: \"Помой раму!\""

    correctly_separated_sentences = ["Мама красиво мыла раму.", "Мама красиво мыла раму...",
                                     "Мама красиво мыла раму!", "Мама красиво мыла раму!!!",
                                     "Мама красиво мыла раму?", "Мама красиво мыла раму?!",
                                     "Мама мыла раму... красиво.", "Мама сказала: \"Помой раму!\""]

    extracted_sentences_from_conllu = [
        {'position': '0',
         'text': 'Красивая - мама красиво, училась в ПДД и ЖКУ '
                 'по адресу Львовская 10 лет с почтой test .',
         'tokens': ['1\tКрасивая\tкрасивый\tADJ\t_\t_\t0\troot\t_\t_',
                    '2\tмама\tмама\tNOUN\t_\t_\t0\troot\t_\t_',
                    '3\tкрасиво\tкрасиво\tADV\t_\t_\t0\troot\t_\t_',
                    '4\tучилась\tучиться\tVERB\t_\t_\t0\troot\t_\t_',
                    '5\tв\tв\tADP\t_\t_\t0\troot\t_\t_',
                    '6\tПДД\tпдд\tNOUN\t_\t_\t0\troot\t_\t_',
                    '7\tи\tи\tCCONJ\t_\t_\t0\troot\t_\t_',
                    '8\tЖКУ\tжку\tNOUN\t_\t_\t0\troot\t_\t_',
                    '9\tпо\tпо\tADP\t_\t_\t0\troot\t_\t_',
                    '10\tадресу\tадрес\tNOUN\t_\t_\t0\troot\t_\t_',
                    '11\tЛьвовская\tльвовский\tADJ\t_\t_\t0\troot\t_\t_',
                    '12\t10\t10\tNUM\t_\t_\t0\troot\t_\t_',
                    '13\tлет\tгод\tNOUN\t_\t_\t0\troot\t_\t_',
                    '14\tс\tс\tADP\t_\t_\t0\troot\t_\t_',
                    '15\tпочтой\tпочта\tNOUN\t_\t_\t0\troot\t_\t_',
                    '16\ttest\ttest\tX\t_\t_\t0\troot\t_\t_',
                    '17\t.\t.\tPUNCT\t_\t_\t0\troot\t_\t_']}]
