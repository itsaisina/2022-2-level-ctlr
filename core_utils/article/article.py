"""
Article implementation
"""
import enum
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol, Sequence

from core_utils.constants import ASSETS_PATH


def date_from_meta(date_txt: str) -> datetime:
    """
    Converts text date to datetime object
    """
    return datetime.strptime(date_txt, "%Y-%m-%d %H:%M:%S")


def get_article_id_from_filepath(path: Path) -> int:
    """
    Extracts the article id from its path
    """
    return int(path.stem.split('_')[0])


def split_by_sentence(text: str) -> list[str]:
    """
    Splits the given text by sentence separators
    """
    pattern = r"(?<!\w\.\w.)(?<![А-Я][а-я]\.)((?<=\.|\?|!)|(?<=\?\"|!\"))\s(?=[А-Я])"
    text = re.sub(r'[\n|\t]+', '. ', text)
    sentences = [sentence for sentence in re.split(pattern, text) if sentence.replace(' ', '')
                 and len(sentence) > 10]
    return sentences


# pylint: disable=too-few-public-methods
class SentenceProtocol(Protocol):
    """
    Protocol definition for sentences to make dependency inversion from direct
    import from lab 6 implementation of ConlluSentence
    """

    def get_cleaned_sentence(self) -> str:
        """
        All tokens should be normalized and joined with a space
        """

    def get_tokens(self) -> list:
        """
        All tokens should be ConlluToken instance
        """

    def get_conllu_text(self, include_morphological_tags: bool) -> str:
        """
        Gets the text in the CONLL-U format
        """


class ArtifactType(enum.Enum):
    """
    Types of artifacts that can be created by text processing pipelines
    """
    CLEANED = 'cleaned'
    MORPHOLOGICAL_CONLLU = 'morphological_conllu'
    POS_CONLLU = 'pos_conllu'


class Article:
    """
    Article class implementation.
    Stores article raw, meta and conllu data
    """

    date: Optional[datetime]
    _conllu_sentences: Sequence[SentenceProtocol]

    def __init__(self, url: Optional[str], article_id: int) -> None:
        self.url = url
        self.article_id = article_id

        self.title = ''
        self.date = None
        self.author = []
        self.topics = []
        self.text = ''
        self.pos_frequencies = {}
        self._conllu_sentences = []

    def set_pos_info(self, pos_freq: dict) -> None:
        """
        Adds POS information in meta file
        """
        self.pos_frequencies = pos_freq

    def get_meta(self) -> dict:
        """
        Gets all meta params
        """
        return {
            'id': self.article_id,
            'url': self.url,
            'title': self.title,
            'date': self._date_to_text() or None,
            'author': self.author,
            'topics': self.topics,
            'pos_frequencies': self.pos_frequencies
        }

    def get_raw_text(self) -> str:
        """
        Gets raw text from the article
        """
        return self.text

    def get_conllu_text(self, include_morphological_tags: bool) -> str:
        """
        Gets the text in the CONLL-U format
        """
        return '\n'.join([sentence.get_conllu_text(include_morphological_tags) for sentence in
                          self._conllu_sentences]) + '\n'

    def set_conllu_sentences(self, sentences: Sequence[SentenceProtocol]) -> None:
        """
        Sets the conllu_sentences_attribute
        """
        self._conllu_sentences = sentences

    def get_conllu_sentences(self) -> Sequence[SentenceProtocol]:
        """
        Returns the sentences from ConlluArticle
        """
        return self._conllu_sentences

    def get_cleaned_text(self) -> str:
        """
        Returns the cleaned text
        """
        return ' '.join([sentence.get_cleaned_sentence() for
                         sentence in self._conllu_sentences])

    def _date_to_text(self) -> str:
        """
        Converts datetime object to text
        """
        return self.date.strftime("%Y-%m-%d %H:%M:%S") if self.date else ''

    def get_raw_text_path(self) -> Path:
        """
        Returns path for requested raw article
        """
        article_txt_name = f"{self.article_id}_raw.txt"
        return ASSETS_PATH / article_txt_name

    def get_meta_file_path(self) -> Path:
        """
        Returns path for requested raw article
        """
        meta_file_name = f"{self.article_id}_meta.json"
        return ASSETS_PATH / meta_file_name

    def get_file_path(self, kind: ArtifactType) -> Path:
        """
        Returns a proper filepath for an Article instance
        kind: variant of a file -- ArtifactType
        """

        conllu = kind in (ArtifactType.POS_CONLLU, ArtifactType.MORPHOLOGICAL_CONLLU)

        extension = '.conllu' if conllu else '.txt'
        article_name = f"{self.article_id}_{kind.value}{extension}"

        return ASSETS_PATH / article_name

    def get_pos_freq(self) -> dict:
        """
        Returns a pos_frequency parameter
        """
        return self.pos_frequencies
