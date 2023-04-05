# pylint: disable=unspecified-encoding
"""
Listing for practice with json module
"""

import json


def main() -> None:
    """
    Entrypoint for a seminar's listing
    """
    # 1. Create JSON file from a dictionary
    student = {'name': 'John', 'surname': 'Davis', 'age': 15, 'hobbies': ['sport', 'reading']}
    res = json.dumps(student)
    with open('sample.json', 'w') as f:
        f.write(res)

    # 2. Write dict to JSON directly
    with open('sample.json', 'w', encoding='utf-8') as f:
        json.dump(student, f, ensure_ascii=True, indent=4, separators=(', ', ': '))

    # 3. Read from JSON
    with open('sample.json', 'r', encoding='utf-8') as f:
        content = json.load(f)

    print(content)


if __name__ == '__main__':
    main()
