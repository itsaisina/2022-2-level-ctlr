"""
Seminar on morphological analysis: pymystem3
"""
# pylint: disable=pointless-string-statement

import time
from pathlib import Path

try:
    from pymystem3 import Mystem
except ImportError:
    print('No libraries installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for module
    """
    mystem = Mystem()

    # news from https://www.nn.ru/text/gorod/2023/04/16/72225200/
    plain_text_path = Path(__file__).parent / '1_raw.txt'

    with plain_text_path.open(encoding='utf-8') as f:
        plain_text = f.read()

    lemmatized_tokens = mystem.lemmatize(plain_text)

    print(type(lemmatized_tokens))  # list

    for token in lemmatized_tokens:
        print(token)

    print(f'Before: {plain_text}')
    print(f'After: {" ".join(lemmatized_tokens)}')

    very_strange_text = '<!@html><body><h1>Hello&@#$ my friend!'
    print(mystem.lemmatize(very_strange_text))

    plain_text_analysis = mystem.analyze(plain_text)
    print(type(plain_text_analysis))

    num_errors = 0
    last_error = None
    for i in plain_text_analysis:
        try:
            morphological_analysis = i['analysis'][0]['gr']
            print(i['analysis'][0]['lex'], morphological_analysis)
        except KeyError:
            num_errors += 1
            last_error = i
    print(f'Error with retrieving information for <{last_error}>')
    print(f'Num failed parses: {num_errors}')

    start = time.time()
    mystem.analyze(plain_text)
    end = time.time() - start
    print(f'As a text, shared instance, took {end:.2} seconds')

    mystem = Mystem()
    start = time.time()
    res = []
    for token in plain_text.split():
        res.append(mystem.analyze(token))
    end = time.time() - start
    print(f'Word by word, shared instance, took {end:.2} seconds')

    start = time.time()
    res = []
    for token in plain_text.split():
        res.append(Mystem().analyze(token))
    end = time.time() - start
    print(f'Word by word, new instance each time, took {end:.2} seconds')

    # Output on lector's machine
    """
    Lecturer's machine results:
        As a text, shared instance, took 0.13 seconds
        Word by word, shared instance, took 1.0 seconds
        Word by word, new instance each time, took 1800 seconds
    """

    # Seminar's tasks:
    # Task 1. Calculate number of nouns
    # Task 1. Cleanup text from any punctuation marks and lowercase it
    # ---
    # Task 1. Find all unique punctuation marks
    # Task 1. Calculate unique punctuation marks
    # Task 1. Are there more nouns than adjectives in a given text?


if __name__ == '__main__':
    main()
