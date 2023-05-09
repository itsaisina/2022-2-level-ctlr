"""
Pipeline for CONLL-U formatting
"""
import re
import string
from itertools import chain
from pathlib import Path
from typing import List

from pymorphy2 import MorphAnalyzer
from pymystem3 import Mystem

from core_utils.article.article import SentenceProtocol, split_by_sentence
from core_utils.article.io import to_cleaned, to_conllu, from_raw
from core_utils.article.ud import OpencorporaTagProtocol, TagConverter

# pylint: disable=too-few-public-methods
from core_utils.constants import ASSETS_PATH


class EmptyDirectoryError(Exception):
    """
    The provided directory is empty
    """
    pass


class InconsistentDatasetError(Exception):
    """
    The dataset is inconsistent,
    such as when the number of raw and meta
    files is not equal, IDs contain slips,
    or files are empty
    """
    pass


class CorpusManager:
    """
    Works with articles and stores them
    """

    def __init__(self, path_to_raw_txt_data: Path):
        """
        Initializes CorpusManager
        """
        self.path_to_raw_txt_data = path_to_raw_txt_data
        self._storage = {}
        self._validate_dataset()
        self._scan_dataset()

    def _validate_dataset(self) -> None:
        """
        Validates folder with assets
        """
        if not self.path_to_raw_txt_data.exists():
            raise FileNotFoundError

        if not self.path_to_raw_txt_data.is_dir():
            raise NotADirectoryError

        file_paths = list(self.path_to_raw_txt_data.glob("*"))
        if not file_paths:
            raise EmptyDirectoryError

        raw_files = sorted(list(self.path_to_raw_txt_data.glob("*_raw.txt")))
        meta_files = sorted(list(self.path_to_raw_txt_data.glob("*_meta.json")))

        if len(raw_files) != len(meta_files):
            raise InconsistentDatasetError

        raw_ind = sorted([int(file.stem.split("_")[0]) for file in raw_files])
        meta_ind = sorted([int(file.stem.split("_")[0]) for file in meta_files])

        for indices in (raw_ind, meta_ind):
            for ind, file_id in enumerate(indices[:-1]):
                if indices[ind + 1] - file_id > 1:
                    raise InconsistentDatasetError

        if any(file.stat().st_size == 0 for file in chain(raw_files, meta_files)):
            raise InconsistentDatasetError

    def _scan_dataset(self) -> None:
        """
        Register each dataset entry
        """
        for raw_file in self.path_to_raw_txt_data.glob("*_raw.txt"):
            article_id = int(re.search(r'\d+', raw_file.stem)[0])
            self._storage[article_id] = from_raw(raw_file)

    def get_articles(self) -> dict:
        """
        Returns storage params
        """
        return self._storage


class MorphologicalTokenDTO:
    """
    Stores morphological parameters for each token
    """

    def __init__(self, lemma: str = "", pos: str = "", tags: str = ""):
        """
        Initializes MorphologicalTokenDTO
        """
        self.lemma = lemma
        self.pos = pos
        self.tags = tags


class ConlluToken:
    """
    Representation of the CONLL-U Token
    """

    def __init__(self, text: str):
        """
        Initializes ConlluToken
        """
        self._text = text
        self.position = 0
        self._morphological_parameters = MorphologicalTokenDTO()

    def set_morphological_parameters(self, parameters: MorphologicalTokenDTO) -> None:
        """
        Stores the morphological parameters
        """
        self._morphological_parameters = parameters

    def get_morphological_parameters(self) -> MorphologicalTokenDTO:
        """
        Returns morphological parameters from ConlluToken
        """
        return self._morphological_parameters

    def get_conllu_text(self, include_morphological_tags: bool) -> str:
        """
        String representation of the token for conllu files
        """
        position = str(self.position)
        text = self._text
        lemma = self._morphological_parameters.lemma
        pos = self._morphological_parameters.pos
        xpos = '_'
        feats = self._morphological_parameters.tags \
            if include_morphological_tags and self._morphological_parameters.tags else '_'
        head = '0'
        deprel = 'root'
        deps = '_'
        misc = '_'

        return '\t'.join([position, text, lemma, pos, xpos, feats, head, deprel, deps, misc])

    def get_cleaned(self) -> str:
        """
        Returns lowercase original form of a token
        """
        return re.sub(r'\W+', '', self._text.lower())


class ConlluSentence(SentenceProtocol):
    """
    Representation of a sentence in the CONLL-U format
    """

    def __init__(self, position: int, text: str, tokens: list[ConlluToken]):
        """
        Initializes ConlluSentence
        """
        self._position = position
        self._text = text
        self._tokens = tokens

    def _format_tokens(self, include_morphological_tags: bool) -> str:
        """
        Creates a formatted string representation of tokens for a sentence
        """
        return "\n".join(token.get_conllu_text(include_morphological_tags) for token in self._tokens)

    def get_conllu_text(self, include_morphological_tags: bool) -> str:
        """
        Creates string representation of the sentence
        """
        sent_id = f"# sent_id = {self._position}\n"
        text = f"# text = {self._text}\n"
        tokens = f"{self._format_tokens(include_morphological_tags)}\n"

        return f"{sent_id}{text}{tokens}"

    def get_cleaned_sentence(self) -> str:
        """
        Returns the lowercase representation of the sentence
        """
        return " ".join(token.get_cleaned() for token in self._tokens if token.get_cleaned()).strip()

    def get_tokens(self) -> list[ConlluToken]:
        """
        Returns sentences from ConlluSentence
        """
        return self._tokens


class MystemTagConverter(TagConverter):
    """
    Mystem Tag Converter
    """

    def convert_morphological_tags(self, tags: str) -> str:  # type: ignore
        """
        Converts the Mystem tags into the UD format
        """
        pos = self.convert_pos(tags)
        tags = re.sub(r'\(.+?\)', lambda match: match.group(0).split('|')[0], tags)
        extracted_tags = re.findall(r'[а-я]+', tags)

        pos_specific_categories = {
            "NOUN": [self.case, self.number, self.gender, self.animacy],
            "VERB": [self.tense, self.number, self.gender],
            "ADJ": [self.case, self.number, self.gender],
            "NUM": [self.case, self.number, self.gender],
            "PRON": [self.case, self.number, self.gender, self.animacy],
        }

        ud_tags = {}

        for tag in extracted_tags:
            for category in pos_specific_categories.get(pos, []):
                if tag in self._tag_mapping[category]:
                    ud_tags[category] = self._tag_mapping[category][tag]

        return '|'.join(f'{k}={v}' for k, v in sorted(ud_tags.items()))

    def convert_pos(self, tags: str) -> str:  # type: ignore
        """
        Extracts and converts the POS from the Mystem tags into the UD format
        """
        extracted_pos = re.search(r'[A-Z]+', tags)[0]
        return self._tag_mapping[self.pos][extracted_pos]


class OpenCorporaTagConverter(TagConverter):
    """
    OpenCorpora Tag Converter
    """

    def convert_pos(self, tags: OpencorporaTagProtocol) -> str:  # type: ignore
        """
        Extracts and converts POS from the OpenCorpora tags into the UD format
        """
        oc_pos = tags.POS
        return self._tag_mapping['POS'].get(oc_pos, self._tag_mapping['POS'].get('UNKN'))

    def convert_morphological_tags(self, tags: OpencorporaTagProtocol) -> str:  # type: ignore
        """
        Converts the OpenCorpora tags into the UD format
        """
        ud_tags = {}
        tag_properties = {
            self.case: tags.case,
            self.number: tags.number,
            self.gender: tags.gender,
            self.animacy: tags.animacy,
        }

        for category, oc_tag in tag_properties.items():
            if oc_tag is not None:
                ud_tag = self._tag_mapping[category].get(oc_tag)
                if ud_tag is not None:
                    ud_tags[category] = ud_tag

        return '|'.join(f'{k}={v}' for k, v in sorted(ud_tags.items()))


class MorphologicalAnalysisPipeline:
    """
    Preprocesses and morphologically annotates sentences into the CONLL-U format
    """

    def __init__(self, corpus_manager: CorpusManager):
        """
        Initializes MorphologicalAnalysisPipeline
        """
        self._corpus = corpus_manager
        self._mystem = Mystem()
        mapping_path = Path(__file__).parent / 'data' / 'mystem_tags_mapping.json'
        self._converter = MystemTagConverter(mapping_path)

    def _process(self, text: str) -> List[ConlluSentence]:
        """
        Returns the text representation as the list of ConlluSentence
        """
        conllu_sentences = []
        result = (i for i in self._mystem.analyze(text))
        for sentence_position, sentence in enumerate(split_by_sentence(text)):
            conllu_tokens = []
            original_sentence = sentence

            tokens = []
            for token in result:
                if token['text'] not in sentence:
                    continue

                sentence = sentence.replace(token['text'], '', 1)
                if any(c.isalnum() for c in token['text']):
                    tokens.append(token)

                if not any(c.isalnum() for c in sentence):
                    break

            tokens.append({'text': '.'})

            for token_position, token in enumerate(tokens, start=1):
                text = token['text']
                if 'analysis' in token and token['analysis']:
                    lex = token['analysis'][0]['lex']
                    pos = self._converter.convert_pos(token['analysis'][0]['gr'])
                    tags = self._converter.convert_morphological_tags(token['analysis'][0]['gr'])
                else:
                    tags, lex = '', text
                    pos = 'PUNCT' if lex in string.punctuation else ('NUM' if lex.isdigit() else 'X')

                conllu_token = ConlluToken(text)
                morph_params = MorphologicalTokenDTO(lex, pos, tags)
                conllu_token.position = token_position
                conllu_token.set_morphological_parameters(morph_params)
                conllu_tokens.append(conllu_token)

            conllu_sentence = ConlluSentence(sentence_position, original_sentence, conllu_tokens)
            conllu_sentences.append(conllu_sentence)
        return conllu_sentences

    def run(self) -> None:
        """
        Performs basic preprocessing and writes processed text to files
        """
        for article in self._corpus.get_articles().values():
            article.set_conllu_sentences(self._process(article.text))
            to_cleaned(article)
            to_conllu(article, include_morphological_tags=False, include_pymorphy_tags=False)
            to_conllu(article, include_morphological_tags=True, include_pymorphy_tags=False)


class AdvancedMorphologicalAnalysisPipeline(MorphologicalAnalysisPipeline):
    """
    Preprocesses and morphologically annotates sentences into the CONLL-U format
    """

    def __init__(self, corpus_manager: CorpusManager):
        """
        Initializes MorphologicalAnalysisPipeline
        """
        super().__init__(corpus_manager)
        self._backup_analyzer = MorphAnalyzer()
        mapping_path = Path(__file__).parent / 'data' / 'opencorpora_tags_mapping.json'
        self._backup_tag_converter = OpenCorporaTagConverter(mapping_path)

    def _process(self, text: str) -> List[ConlluSentence]:
        """
        Returns the text representation as the list of ConlluSentence
        """
        conllu_sentences = []
        result = (i for i in self._mystem.analyze(text))
        for sentence_position, sentence in enumerate(split_by_sentence(text)):
            conllu_tokens = []
            original_sentence = sentence

            tokens = []
            for token in result:
                if token['text'] not in sentence:
                    continue

                sentence = sentence.replace(token['text'], '', 1)
                if any(c.isalnum() for c in token['text']):
                    tokens.append(token)

                if not any(c.isalnum() for c in sentence):
                    break

            tokens.append({'text': '.'})

            for token_position, token in enumerate(tokens, start=1):
                text = token['text']
                if 'analysis' in token and token['analysis']:
                    pos = self._converter.convert_pos(token['analysis'][0]['gr'])
                    if pos == 'NOUN':
                        lex = self._backup_analyzer.parse(text)[0].normal_form
                        open_corpora_tags = self._backup_analyzer.parse(text)[0].tag
                        pos = self._backup_tag_converter.convert_pos(open_corpora_tags)
                        tags = self._backup_tag_converter.convert_morphological_tags(open_corpora_tags)
                    else:
                        lex = token['analysis'][0]['lex']
                        tags = self._converter.convert_morphological_tags(token['analysis'][0]['gr'])
                else:
                    tags, lex = '', text
                    pos = 'PUNCT' if lex in string.punctuation else ('NUM' if lex.isdigit() else 'X')

                conllu_token = ConlluToken(text)
                morph_params = MorphologicalTokenDTO(lex, pos, tags)
                conllu_token.position = token_position
                conllu_token.set_morphological_parameters(morph_params)
                conllu_tokens.append(conllu_token)

            conllu_sentence = ConlluSentence(sentence_position, original_sentence, conllu_tokens)
            conllu_sentences.append(conllu_sentence)
        return conllu_sentences

    def run(self) -> None:
        """
        Performs basic preprocessing and writes processed text to files
        """
        for article in self._corpus.get_articles().values():
            article.set_conllu_sentences(self._process(article.text))
            to_conllu(article, include_morphological_tags=True, include_pymorphy_tags=True)


def main() -> None:
    """
    Entrypoint for pipeline module
    """
    corpus_manager = CorpusManager(ASSETS_PATH)
    pipe = MorphologicalAnalysisPipeline(corpus_manager)
    pipe.run()
    advanced_pipe = AdvancedMorphologicalAnalysisPipeline(corpus_manager)
    advanced_pipe.run()


if __name__ == "__main__":
    main()
