"""
Tests for student text preprocessing
"""
import re
import shutil
import unittest

import pytest

from config.test_params import TEST_PATH
from core_utils.article.ud import extract_sentences_from_raw_conllu
from lab_6_pipeline.pipeline import CorpusManager
from lab_6_pipeline.tests.utils import pipeline_setup

UD_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
           'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
PUNCTUATION_MARKS = [',', '.', '-', ';', ':', '!', '?', '<']


class StudentTextBasicPreprocessTest(unittest.TestCase):
    """
    Class for checking basic text preprocessing
    """

    def setUp(self) -> None:
        pipeline_setup()
        CorpusManager(TEST_PATH)

        self.articles = {}
        for file in TEST_PATH.iterdir():
            if file.name.endswith("_cleaned.txt"):
                with file.open("r", encoding="utf-8") as txt:
                    self.articles[int(file.name[:-12])] = txt.read()

    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_5_student_dataset_validation
    @pytest.mark.lab_6_pipeline
    def test_clean_tokens(self):
        """
        Ensure there is no punctuation of uppercase in cleaned text
        """
        for article_id, article_text in self.articles.items():
            for token in article_text.split():
                message = f"There are some punctuation " \
                          f"marks found in article {article_id}"
                self.assertTrue(token not in PUNCTUATION_MARKS, message)
                if token.isnumeric():
                    continue
                message = f"Token {token} in article " \
                          f"{article_id} is not lowercase"
                self.assertTrue(token.islower(), msg=message)

    def tearDown(self) -> None:
        shutil.rmtree(TEST_PATH)


class StudentTextAdvancedPreprocessTest(unittest.TestCase):
    """
    Class for checking advanced text preprocessing
    """

    def setUp(self) -> None:
        pipeline_setup()
        CorpusManager(TEST_PATH)

        self.articles = {}
        for file in TEST_PATH.iterdir():
            if file.name.endswith(".conllu"):
                with file.open("r", encoding="utf-8") as txt:
                    self.articles[int(file.name.split('_')[0])] = \
                        extract_sentences_from_raw_conllu(txt.read())

    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_5_student_dataset_validation
    @pytest.mark.lab_6_pipeline
    def test_conllu_tokens_format(self):
        """
        Ensure conllu sequences are formatted correctly
        """
        for sentence_id, sentences in self.articles.items():
            for sentence in sentences:
                message = f"There should be # sent_id = some-number in sentence №{sentence_id}"
                self.assertTrue(sentence.get('position', False), msg=message)

                message = f"There should be # text = some-text in sentence №{sentence_id}"
                self.assertTrue(sentence.get('text', False), msg=message)

                message = f'There should be tokens in sentence №{sentence_id}'
                self.assertTrue(sentence.get('tokens', False), msg=message)

    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_5_student_dataset_validation
    @pytest.mark.lab_6_pipeline
    def test_ud_tags_correctness_conllu(self):
        """
        Ensure there are no unknown UD tags in multi-tagged sequences
        """
        tags_pattern = re.compile(r"\t\w+\t\w+\t([A-Z]+)\t")
        for _, sentences in self.articles.items():
            for sentence in sentences:
                for tokens_line in sentence['tokens']:
                    tags = re.findall(tags_pattern, tokens_line)
                    for tag in tags:
                        message = f"Tag {tag} not in list of known mystem tags"
                        self.assertTrue(tag in UD_TAGS, msg=message)

    def tearDown(self) -> None:
        shutil.rmtree(TEST_PATH)
