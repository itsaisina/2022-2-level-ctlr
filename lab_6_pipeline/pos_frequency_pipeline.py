"""
Implementation of POSFrequencyPipeline for score ten only.
"""
from collections import Counter
from pathlib import Path
from typing import Optional

from core_utils.article.article import Article, ArtifactType, get_article_id_from_filepath
from core_utils.article.io import to_meta, from_meta
from core_utils.article.ud import extract_sentences_from_raw_conllu
from core_utils.constants import ASSETS_PATH
from core_utils.visualizer import visualize
from lab_6_pipeline.pipeline import ConlluToken, ConlluSentence, CorpusManager, MorphologicalTokenDTO


def from_conllu(path: Path, article: Optional[Article] = None) -> Article:
    """
    Populates the Article abstraction with all information from the conllu file
    """
    with open(path, encoding='utf-8') as f:
        content = f.read()

    extracted_sentences = extract_sentences_from_raw_conllu(content)
    for sentence in extracted_sentences:
        sentence['tokens'] = [_parse_conllu_token(token) for token in sentence['tokens']]

    conllu_sentences = [ConlluSentence(**sentence) for sentence in extracted_sentences]

    if not article:
        article_id = get_article_id_from_filepath(path)
        article = Article(None, article_id)

    article.set_conllu_sentences(conllu_sentences)
    return article


def _parse_conllu_token(token_line: str) -> ConlluToken:
    """
    Parses the raw text in the CONLLU format into the CONLL-U token abstraction
    Example:
    '2	произошло	происходить	VERB	_	Gender=Neut|Number=Sing|Tense=Past	0	root	_	_'
    """
    params = token_line.split('\t')

    token = ConlluToken(text=params[1])
    token.set_position(position=int(params[0]))
    morph_params = MorphologicalTokenDTO(lemma=params[2],
                                         pos=params[3],
                                         tags=params[5])
    token.set_morphological_parameters(morph_params)

    return token


# pylint: disable=too-few-public-methods
class EmptyFileError(Exception):
    """
    Raises when a file is found to be empty
    """
    pass


class POSFrequencyPipeline:
    """
    Counts frequencies of each POS in articles,
    updates meta information and produces graphic report
    """

    def __init__(self, corpus_manager: CorpusManager):
        """
        Initializes PosFrequencyPipeline
        """
        self.corpus_manager = corpus_manager

    def run(self) -> None:
        """
        Visualizes the frequencies of each part of speech
        """
        articles = self.corpus_manager.get_articles().values()
        for article in articles:
            conllu_path = article.get_file_path(ArtifactType.MORPHOLOGICAL_CONLLU)

            if conllu_path.stat().st_size == 0:
                raise EmptyFileError

            processed_article = from_meta(article.get_meta_file_path(), from_conllu(conllu_path))
            pos_freq = self._count_frequencies(processed_article)
            article.set_pos_info(pos_freq)
            to_meta(article)

            image_path = ASSETS_PATH / f"{article.article_id}_image.png"
            visualize(article=article, path_to_save=image_path)

    def _count_frequencies(self, article: Article) -> dict[str, int]:
        """
        Counts POS frequency in Article
        """
        pos_freq = Counter()
        for sentence in article.get_conllu_sentences():
            tokens = sentence.get_tokens()
            for token in tokens:
                pos = token.get_morphological_parameters().pos
                pos_freq[pos] += 1
        return pos_freq


def main() -> None:
    """
    Entrypoint for the module
    """
    corpus_manager = CorpusManager(ASSETS_PATH)
    pipeline = POSFrequencyPipeline(corpus_manager=corpus_manager)
    pipeline.run()


if __name__ == "__main__":
    main()
