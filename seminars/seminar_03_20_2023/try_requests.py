# pylint: disable=missing-timeout
"""
Listing for practice with requests library
"""
import random
import time

try:
    import requests
    from requests import HTTPError, Timeout
except ImportError:
    print('No libraries installed. Failed to import.')


def main() -> None:
    """
    Module entrypoint
    """

    # 1. requests basics
    correct_url = 'https://pypi.org/project/requests/'
    incorrect_url = f'{correct_url}garbagegarbage'

    response = requests.get(correct_url)
    if response:
        print(f'Response code is: {response.status_code}')

    if response.ok:
        print('Response is OK')

    try:
        response = requests.get(incorrect_url)
        response.raise_for_status()
    except HTTPError as e:
        print(f'Error occurred: {e.response.status_code}')

    # 2. requests advanced
    # 2.1 handling timeouts, need to specify it for long responding servers
    try:
        response = requests.get(incorrect_url, timeout=0.000001)
    except Timeout as e:
        print(f'In timeout: {e}')

    # 2.2 making pauses between requests to make them look more natural for a server
    response = requests.get(correct_url)

    sleep_period = 10
    print(f'Sleeping for {sleep_period}')
    time.sleep(sleep_period)

    response = requests.get(correct_url)

    sleep_period = random.randrange(3, 7)
    print(f'Sleeping for {sleep_period}')
    time.sleep(sleep_period)

    response = requests.get(correct_url)

    # 2.3 specifying browser headers to make a request look more natural for a server
    response = requests.get(correct_url)
    print(response.request.headers)
    print(response.headers)

    response = requests.get(correct_url, headers={
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/111.0.0.0 Safari/537.36'
    })
    print(response.request.headers)
    print(response.headers)

    # 3. working with responses
    # 3.1 getting HTML page content as a plain Python string
    response = requests.get(correct_url)
    print(response.text)

    # 3.2 saving HTML page as a file
    response = requests.get('https://www.nn.ru/text/gorod/2023/03/19/72141902/')

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(response.text)

    # 3.3 saving binary files formats, such as images
    response = requests.get('https://pypi.org/static/images/logo-small.2a411bc6.svg')

    with open('logo.png', 'wb') as f:
        f.write(response.content)


if __name__ == '__main__':
    main()
