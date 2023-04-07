# pylint: disable=R0801
"""
Seminar that kicks-off HTMLParser

1. GET request to article page
2. Create BeautifulSoup instance on top of response
3. Find Title + all article text

------- BELOW IS OPTIONAL
4. Save it to file <PROJECT_ROOT>/tmp/data/1.txt

5. Find Title and author of the Article # mark 6

6. Find date of the Article # mark 8
"""

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print('No libraries installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for module
    """
    url = 'https://www.nn.ru/text/education/2023/04/06/72194864/'
    response = requests.get(url, timeout=5)
    print(response.status_code)

    main_bs = BeautifulSoup(response.text, 'lxml')

    # Idea no. 1: by tag name
    # Idea no. 2: by class name
    # Idea no. 3: by parent
    # Idea no. 4: by XPath (very-very-very unlikely)
    # Idea no. 5: by attributes (id, ...)

    # only first or None
    # title_bs = main_bs.find('h1', class_='central-right-wrapper')

    # []
    title_bs = main_bs.find_all('h1', {'itemprop': 'headline'})[0]
    print(title_bs, type(title_bs))

    span_bs = title_bs.find('span')
    print(span_bs.text)

    body_bs = main_bs.find_all('div', {'itemprop': 'articleBody'})[0]

    all_paragraphs = body_bs.find_all('p')
    print(len(all_paragraphs))


if __name__ == '__main__':
    main()
