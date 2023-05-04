"""
Pipeline for CONLL-U formatting
"""
import re
import string
from pathlib import Path
from typing import List

from pymorphy2 import MorphAnalyzer
from pymystem3 import Mystem

from core_utils.article.article import SentenceProtocol, Article, split_by_sentence
from core_utils.article.io import to_cleaned, to_conllu
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
        self._validate_dataset()
        self._storage = {}
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

        raw_files = [file for file in file_paths if file.suffix == ".txt" and file.stem.endswith("_raw")]
        meta_files = [file for file in file_paths if file.suffix == ".json" and file.stem.endswith("_meta")]

        if len(raw_files) != len(meta_files):
            raise InconsistentDatasetError

        raw_ind = sorted([int(file.stem.split("_")[0]) for file in raw_files])
        for ind, file_id in enumerate(raw_ind[:-1]):
            if raw_ind[ind + 1] - file_id > 1:
                raise InconsistentDatasetError

        for raw_file in raw_files:
            if raw_file.stat().st_size == 0:
                raise InconsistentDatasetError

    def _scan_dataset(self) -> None:
        """
        Register each dataset entry
        """
        for raw_file in self.path_to_raw_txt_data.glob("*_raw.txt"):
            article_id = int(raw_file.stem.split("_")[0])
            article = Article(url=None, article_id=article_id)

            with open(raw_file, 'r', encoding='utf-8') as f:
                article.text = f.read()

            self._storage[article_id] = article

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

    def set_position(self, position: int) -> None:
        self.position = position

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
        return self._text.lower().translate(str.maketrans('', '', string.punctuation + "«»–"))


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
        return " ".join(token.get_cleaned() for token in self._tokens).replace('  ', ' ').strip()

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
        ud_tags = {}

        pos_specific_categories = {
            "NOUN": [self.case, self.number, self.gender, self.animacy],
            "VERB": [self.tense, self.number, self.gender],
            "ADJ": [self.case, self.number, self.gender],
            "ADV": [],
            "CCONJ": [],
            "INTJ": [],
            "NUM": [self.case, self.number, self.gender],
            "PART": [],
            "ADP": [],
            "PRON": [self.case, self.number, self.gender, self.animacy],
        }

        for tag in extracted_tags:
            for category in pos_specific_categories.get(pos, []):
                if tag in self._tag_mapping[category]:
                    ud_tags[category] = self._tag_mapping[category][tag]

        return '|'.join(f'{k}={v}' for k, v in sorted(ud_tags.items()))

    def convert_pos(self, tags: str) -> str:  # type: ignore
        """
        Extracts and converts the POS from the Mystem tags into the UD format
        """
        extracted_pos = re.search(r'[A-Z]+', tags)
        return self._tag_mapping["POS"].get(extracted_pos[0], "")


class OpenCorporaTagConverter(TagConverter):
    """
    OpenCorpora Tag Converter
    """

    def convert_pos(self, tags: OpencorporaTagProtocol) -> str:  # type: ignore
        """
        Extracts and converts POS from the OpenCorpora tags into the UD format
        """
        for oc_tag, ud_tag in self._tag_mapping['POS'].items():
            if oc_tag in tags:
                return ud_tag
        return 'UNKN'

    def convert_morphological_tags(self, tags: OpencorporaTagProtocol) -> str:  # type: ignore
        """
        Converts the OpenCorpora tags into the UD format
        """
        ud_tags = {}
        for category in ('Case', 'Number', 'Gender', 'Animacy'):
            for oc_tag, ud_tag in self._tag_mapping[category].items():
                if oc_tag in tags:
                    ud_tags[category] = ud_tag
                    break
        return '|'.join(f'{k}={v}' for k, v in sorted(ud_tags.items()))


class MorphologicalAnalysisPipeline:
    """
    Preprocesses and morphologically annotates sentences into the CONLL-U format
    """

    def __init__(self, corpus_manager: CorpusManager):
        """
        Initializes MorphologicalAnalysisPipeline
        """
        path = Path(__file__).parent / 'data' / 'mystem_tags_mapping.json'
        self.corpus_manager = corpus_manager
        self.tag_converter = MystemTagConverter(path)

    def _process(self, text: str) -> List[ConlluSentence]:
        """
        Returns the text representation as the list of ConlluSentence
        """
        mystem = Mystem()
        mystem_result = mystem.analyze(text)

        alphanumeric_mystem_result = list(filter(lambda x: x['text'].isalnum(), mystem_result))

        sentences = []

        result_idx = 0
        for sentence_position, sentence in enumerate(split_by_sentence(text)):
            sentence_tokens = []
            tokens_with_punctuations = re.findall(r'\w+|[.!?]+', sentence)
            token_position = 1
            for token in tokens_with_punctuations:
                conllu_token = ConlluToken(token)

                if token.isalnum() and result_idx < len(alphanumeric_mystem_result):
                    mystem_token = alphanumeric_mystem_result[result_idx]

                    if 'analysis' in mystem_token and mystem_token['analysis']:
                        lex = mystem_token['analysis'][0]['lex']
                        pos = self.tag_converter.convert_pos(mystem_token['analysis'][0]['gr'])
                        tags = self.tag_converter.convert_morphological_tags(mystem_token['analysis'][0]['gr'])
                    elif mystem_token['text'].isdigit():
                        lex = mystem_token['text']
                        pos = 'NUM'
                        tags = ''
                    else:
                        lex = mystem_token['text']
                        pos = 'X'
                        tags = ''

                    result_idx += 1
                else:
                    lex = token
                    pos = 'PUNCT'
                    tags = ''

                morph_params = MorphologicalTokenDTO(lex, pos, tags)
                conllu_token.set_position(token_position)
                conllu_token.set_morphological_parameters(morph_params)
                sentence_tokens.append(conllu_token)
                token_position += 1

            sentence = ConlluSentence(sentence_position, sentence, sentence_tokens)
            sentences.append(sentence)

        return sentences

    def run(self) -> None:
        """
        Performs basic preprocessing and writes processed text to files
        """
        for article in self.corpus_manager.get_articles().values():
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
        mystem = Mystem()
        mystem_result = mystem.analyze(text)

        alphanumeric_mystem_result = list(filter(lambda x: x['text'].isalnum(), mystem_result))

        sentences = []

        result_idx = 0
        for sentence_position, sentence in enumerate(split_by_sentence(text)):
            sentence_tokens = []
            tokens_with_punctuations = re.findall(r'\w+|[.!?]+', sentence)
            token_position = 1
            for token in tokens_with_punctuations:
                conllu_token = ConlluToken(token)

                if token.isalnum():
                    mystem_token = alphanumeric_mystem_result[result_idx]

                    if 'analysis' in mystem_token and mystem_token['analysis']:
                        pos = self.tag_converter.convert_pos(mystem_token['analysis'][0]['gr'])
                        if pos == 'NOUN':
                            lex = self._backup_analyzer.parse(mystem_token['text'])[0].normal_form
                            pos = self._backup_tag_converter.convert_pos(
                                self._backup_analyzer.parse(mystem_token['text'])[0].tag)
                            tags = self._backup_tag_converter.convert_morphological_tags(
                                self._backup_analyzer.parse(mystem_token['text'])[0].tag)
                        else:
                            tags = self.tag_converter.convert_morphological_tags(mystem_token['analysis'][0]['gr'])
                            lex = mystem_token['analysis'][0]['lex']
                    elif mystem_token['text'].isdigit():
                        lex = mystem_token['text']
                        pos = 'NUM'
                        tags = ''
                    else:
                        lex = mystem_token['text']
                        pos = 'X'
                        tags = ''

                    result_idx += 1
                else:
                    lex = token
                    pos = 'PUNCT'
                    tags = ''

                morph_params = MorphologicalTokenDTO(lex, pos, tags)
                conllu_token.set_position(token_position)
                conllu_token.set_morphological_parameters(morph_params)
                sentence_tokens.append(conllu_token)
                token_position += 1

            sentence = ConlluSentence(sentence_position, sentence, sentence_tokens)
            sentences.append(sentence)

        return sentences

    def run(self) -> None:
        """
        Performs basic preprocessing and writes processed text to files
        """
        for article in self.corpus_manager.get_articles().values():
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
