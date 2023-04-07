"""
Parsers for CONLL-U
"""
import json
import re
from pathlib import Path
from typing import Protocol, Union


class OpencorporaTagProtocol(Protocol):
    """
    Abstraction definition for pymorphy2.tagset.OpencorporaTag
    Link: https://pymorphy2.readthedocs.io/en/stable/_modules/pymorphy2/tagset.html
    """

    # pylint: disable=invalid-name
    @property
    def POS(self) -> str:
        """
        pymorphy2 attribute: POS
        """

    @property
    def gender(self) -> str:
        """
        pymorphy2 attribute: gender
        """

    @property
    def number(self) -> str:
        """
        pymorphy2 attribute: number
        """

    @property
    def animacy(self) -> str:
        """
        pymorphy2 attribute: animacy
        """

    @property
    def case(self) -> str:
        """
        pymorphy2 attribute: case
        """

    @property
    def tense(self) -> str:
        """
        pymorphy2 attribute: tense
        """


def extract_sentences_from_raw_conllu(conllu_article_text: str) -> list[dict]:
    """
    Extracts sentences from the CONLL-U-formatted article and stores them like:

    [
        {
            'position': sentence_position,
            'text': sentence_text,
            'tokens': sentence_tokens
        },
        {
            'position': sentence_position,
            'text': sentence_text,
            'tokens': sentence_tokens
        },
        ...
    ]
    """
    sentences = []
    parts = re.split(r'(#\ssent_id\s=\s\d+\n#\stext\s=\s.+)\n', conllu_article_text)[1:]
    for part_id in range(0, len(parts), 2):
        sentence = {'position': re.search(r'#\ssent_id\s=\s(\d+)', parts[part_id]).group(1),
                    'text': re.search(r'#\stext\s=\s(.+)', parts[part_id]).group(1),
                    'tokens': parts[part_id + 1].split('\n')}
        sentence['tokens'] = [token for token in sentence['tokens'] if token]
        sentences.append(sentence)
    return sentences


class TagConverter:
    """
    Tag Converter Abstraction
    """
    _tag_mapping: dict[str, dict[str, str]]

    def __init__(self, tag_mapping_path: Path):
        """
        Initializes Converter
        """
        with open(tag_mapping_path, 'r', encoding='utf-8') as mapping_file:
            self._tag_mapping = json.load(mapping_file)

        self.pos = 'POS'
        self.case = 'Case'
        self.number = 'Number'
        self.gender = 'Gender'
        self.animacy = 'Animacy'
        self.tense = 'Tense'
        self.tags = 'TAGS'

    def convert_morphological_tags(self, tags: Union[str, OpencorporaTagProtocol]) -> str:
        """
        Converts the tags into the UD format
        """
        raise NotImplementedError

    def convert_pos(self, tags: Union[str, OpencorporaTagProtocol]) -> str:
        """
        Extracts and converts POS from the tags into the UD format
        """
        raise NotImplementedError
