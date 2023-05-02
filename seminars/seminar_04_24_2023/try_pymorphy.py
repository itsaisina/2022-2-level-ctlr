"""
Seminar on morphological analysis: pymorphy
"""
# pylint: disable=pointless-string-statement

import time
from pathlib import Path

try:
    import pymorphy2
except ImportError:
    print('No libraries installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for module
    """
    morph_analyzer = pymorphy2.MorphAnalyzer()
    all_parses = morph_analyzer.parse('стали')
    print(f'Analyzer found {len(all_parses)} different options of what this word means')

    # Usually we should take the first one - it is correct for most of the cases
    parsing_result = morph_analyzer.parse('стали')[0]

    # Parsing result has a Tag object, to write it to file,
    # it should be converted to string first of all
    print(parsing_result.tag)

    # Inspect Tag object as it has many important attributes
    print(parsing_result.tag.POS)

    # If you do not understand English terms of morphological analysis, use Russian translation
    print(parsing_result.tag.cyr_repr)

    # To get just a normal form, use an attribute `normal_form`
    print(parsing_result.normal_form)

    # To get full Parse object for a normal form use another property: `normalized`
    print(parsing_result.normalized)

    plain_text_path = Path(__file__).parent / '1_raw.txt'

    with plain_text_path.open(encoding='utf-8') as f:
        plain_text = f.read()

    all_words = plain_text.split()

    start = time.time()
    morph_result = {}
    for word in all_words:
        morph_result[word] = pymorphy2.MorphAnalyzer().parse(word)[0].tag
    many_instances_time = time.time() - start

    start = time.time()
    morph_analyzer = pymorphy2.MorphAnalyzer()
    morph_result = {}
    for word in all_words:
        morph_result[word] = morph_analyzer.parse(word)[0].tag
    single_instance_time = time.time() - start

    print(
        f'Time spent (seconds) for MorphAnalyzer instance per each word: {many_instances_time:.2f}')
    print(
        'Time spent (seconds) for MorphAnalyzer instance per each word: '
        f'{single_instance_time:.2f}'
    )  # 0.1 sec

    print(
        f'Single instance is quicker in {many_instances_time / single_instance_time: .2f}x')  # 41x
    """
    Lecturer's machine results:
        Time spent (seconds) for MorphAnalyzer instance per each word: 196.82
        Time spent (seconds) for MorphAnalyzer instance per each word: 1.11
        Single instance is quicker in  177.69x
    """
    # Very interesting to read performance comparison:
    # https://pymorphy2.readthedocs.io/en/stable/internals/dict.html#id13


if __name__ == '__main__':
    main()
