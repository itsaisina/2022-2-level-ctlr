"""
Pipeline for CONLL-U formatting
"""
import string
from pathlib import Path
from typing import List

from pymystem3 import Mystem

from core_utils.article.article import SentenceProtocol, Article
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

    def get_morphological_parameters(self) -> MorphologicalTokenDTO:
        """
        Returns morphological parameters from ConlluToken
        """
        return self._morphological_parameters

    def get_conllu_text(self, include_morphological_tags: bool) -> str:
        """
        String representation of the token for conllu files
        """
        pos = self._morphological_parameters.pos
        lemma = self._morphological_parameters.lemma
        position = self.position
        text = self._text
        xpos = "_"
        feats = "_" if not include_morphological_tags else self._morphological_parameters.tags
        head = "0"
        deprel = "root"
        deps = "_"
        misc = "_"
        return f"{position}\t{text}\t{lemma}\t{pos}\t{xpos}\t{feats}\t{head}\t{deprel}\t{deps}\t{misc}"

    def get_cleaned(self) -> str:
        """
        Returns lowercase original form of a token
        """
        return self._text.lower().translate(str.maketrans('', '', string.punctuation))


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
        tokens = self._format_tokens(include_morphological_tags)
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
        ud_tags = []
        for tag in tags.split(','):
            for key, values in self._tag_mapping.items():
                if tag in values:
                    ud_tags.append(f"{key}={values[tag]}")
                    break
        return "|".join(ud_tags)

    def convert_pos(self, tags: str) -> str:  # type: ignore
        """
        Extracts and converts the POS from the Mystem tags into the UD format
        """
        for mystem_tag, ud_tag in self._tag_mapping['POS'].items():
            if mystem_tag in tags:
                return ud_tag
        return 'X'


class OpenCorporaTagConverter(TagConverter):
    """
    OpenCorpora Tag Converter
    """

    def convert_pos(self, tags: OpencorporaTagProtocol) -> str:  # type: ignore
        """
        Extracts and converts POS from the OpenCorpora tags into the UD format
        """

    def convert_morphological_tags(self, tags: OpencorporaTagProtocol) -> str:  # type: ignore
        """
        Converts the OpenCorpora tags into the UD format
        """


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

        sentences = []
        sentence_position = 1
        sentence_tokens = []

        for word_info in mystem_result:
            if word_info.get("analysis"):
                analysis = word_info["analysis"][0]
                lemma = analysis["lex"]
                mystem_pos = analysis["gr"].split(",", 1)[0]
                ud_pos = self.tag_converter.convert_pos(mystem_pos)

                morph_params = MorphologicalTokenDTO(lemma, ud_pos)
                token = ConlluToken(word_info["text"].replace(',', '.'))
                token.set_morphological_parameters(morph_params)
                sentence_tokens.append(token)
            elif word_info["text"].isdigit():
                morph_params = MorphologicalTokenDTO(word_info["text"], 'NUM')
                token = ConlluToken(word_info["text"].replace(',', '.'))
                token.set_morphological_parameters(morph_params)
                sentence_tokens.append(token)
            else:
                morph_params = MorphologicalTokenDTO(word_info["text"], 'X')
                token = ConlluToken(word_info["text"].replace(',', '.'))
                token.set_morphological_parameters(morph_params)
                sentence_tokens.append(token)

            if word_info["text"] == "\n":
                if sentence_tokens:
                    sentences.append(
                        ConlluSentence(sentence_position, " ".join(t.get_cleaned() for t in sentence_tokens),
                                       sentence_tokens))
                    sentence_tokens = []
                    sentence_position += 1

        if sentence_tokens:
            sentences.append(
                ConlluSentence(sentence_position, " ".join(t.get_cleaned() for t in sentence_tokens), sentence_tokens))

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

    def _process(self, text: str) -> List[ConlluSentence]:
        """
        Returns the text representation as the list of ConlluSentence
        """

    def run(self) -> None:
        """
        Performs basic preprocessing and writes processed text to files
        """


def main() -> None:
    """
    Entrypoint for pipeline module
    """
    manager = CorpusManager(ASSETS_PATH)
    morph_pipe = MorphologicalAnalysisPipeline(manager)
    morph_pipe.run()
    advanced_morph_pipe = AdvancedMorphologicalAnalysisPipeline(manager)
    advanced_morph_pipe.run()


if __name__ == "__main__":
    main()
