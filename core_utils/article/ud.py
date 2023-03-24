"""
Parsers for CONLL-U
"""
import re

try:
    from lab_6_pipeline.conllu import ConlluToken  # type: ignore
    from lab_6_pipeline.conllu import MorphologicalTokenDTO  # type: ignore
    from lab_6_pipeline.conllu import SyntacticTokenDTO  # type: ignore
except ImportError:
    ConlluToken = None
    MorphologicalTokenDTO = None
    SyntacticTokenDTO = None
    print('Unable to import: lab_6_pipeline.conllu.ConlluToken,'
          'MorphologicalTokenDTO,SyntacticTokenDTO')


def parse_conllu_token(token_line: str) -> ConlluToken:
    """
    Parses the raw text in the CONLLU format into the CONLL-U token abstraction

    Example:
        '2\tпроизошло\tпроисходить\tVERB\t_\tGender=Neut|Number=Sing|Tense=Past\t0\troot\t_\t_'
    """

    (position, token_text, lemma, pos, _,
     morphological_tags, parent_position,
     dependency, _, misc) = token_line.split('\t')

    conllu_token = ConlluToken(text=token_text)
    morphological_parameters = MorphologicalTokenDTO(lemma=lemma,
                                                     pos=pos,
                                                     tags=morphological_tags,
                                                     misc=misc)
    syntactic_parameters = SyntacticTokenDTO(position=position,
                                             dependency=dependency,
                                             parent_position=parent_position)

    conllu_token.set_morphological_parameters(morphological_parameters)
    conllu_token.set_syntactic_parameters(syntactic_parameters)

    return conllu_token


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
