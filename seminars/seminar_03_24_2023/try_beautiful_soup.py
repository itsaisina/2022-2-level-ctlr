# pylint: disable=missing-timeout
"""
Listing for practice with beautifulsoup4 library
"""

from urllib.parse import urlparse, urlunparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print('No libraries installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for a seminar's listing
    """
    correct_url = 'https://www.nn.ru/text/gorod/2023/04/03/72186935/'
    response = requests.get(correct_url)

    print(response.text)

    # 1. Creating instance of soup
    # install 'lxml' first or remove it from arguments below
    soup = BeautifulSoup(response.text, 'lxml')

    # 2. Getting tags by dot notation
    print(soup.title)
    print(type(soup.title))
    print(type(soup.title.text))

    # 3. Finding tags by their name
    all_spans = soup.find_all('span')
    print(f'Number of spans: {len(all_spans)}')

    # 4. Finding elements by their class
    header = soup.find_all(class_='_3Esly')
    if header:
        print(f'Found a header: {header}')
    else:
        print('Header not found')

    # 5. Finding elements by their id
    header = soup.find_all(id='record-header')
    if header:
        print(f'Found {len(header)} header(s) by ID: "record-header"')
    else:
        print('Header not found')

    # 6. You can mix them all if you need
    rating = soup.find_all('section', class_='_12gEL _2XsA2')
    if rating:
        print(f'Found a rating string: {rating}')
        print(rating[0].p.text)

    # 7. Find by custom attribute
    all_body = soup.find_all('div', itemprop='articleBody')

    texts = []
    if all_body:
        all_divs = all_body[0].find_all('div')
        texts = []
        for div_bs in all_divs:
            texts.append(div_bs.text)

    print('All text from a page:')
    print(' '.join(texts))

    # 8. Find any link by tag and get its attributes
    all_links = soup.find_all('a')
    for link in all_links:
        try:
            address = link['href']
        except KeyError:
            continue
        parsed_address = urlparse(address)
        print(f'Parsing the URL: {address}. '
              f'Protocol: {parsed_address.scheme}. '
              f'Netloc: {parsed_address.netloc}.')
        print(f'\tPath: {parsed_address.path}. Params: {parsed_address.params}.')

        if not parsed_address.netloc:
            print('This is a relative path. Let us construct the full path.')
            full_url = urlunparse((
                urlparse(correct_url).scheme,
                urlparse(correct_url).netloc,
                parsed_address.path,
                None,
                None,
                None
            ))
            print(f'And it is: {full_url}')

        # skipping all other links - remove break if you want all links to be processed
        break


if __name__ == '__main__':
    main()
