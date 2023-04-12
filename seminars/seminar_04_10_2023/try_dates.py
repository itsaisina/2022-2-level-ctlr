# pylint: disable=R0801,too-many-locals
"""
Lecture on dates
"""
from datetime import datetime

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print('No libraries installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for module
    """
    start_date = datetime(2023, 4, 10, 15, 10, 00)
    print(start_date)

    # 1. Construct datetime instance from string
    # 6 апреля 2023, 10:00
    article_date_raw = '6.04.2023'
    article_date = datetime.strptime(
        article_date_raw,
        '%d.%m.%Y'
    )
    print(article_date)

    article_date_raw = '6 April 23'
    article_date = datetime.strptime(
        article_date_raw,
        '%d %B %y'
    )
    print(article_date)

    article_date_raw = '6 апр 23'
    print(f'Before replacement: {article_date_raw}')
    months = {
        'янв': '01',
        # other months are going here... lecturer is lazy
        'апр': '04'
    }
    for month_name, month_number in months.items():
        if month_name in article_date_raw:
            article_date_raw = article_date_raw.replace(
                month_name,
                month_number
            )
    print(f'After replacement: {article_date_raw}')
    article_date = datetime.strptime(
        article_date_raw,
        '%d %m %y'
    )
    print(article_date)

    # 2. Represent datetime in unified format
    final_date = datetime.strftime(article_date, '%Y:%m day: %d')
    print(final_date)

    url = 'https://www.nn.ru/text/education/2023/04/06/72194864/'
    response = requests.get(url, timeout=3)
    main_bs = BeautifulSoup(response.text, 'lxml')
    title_bs = main_bs.find('time')
    date_raw = str(title_bs.get('datetime'))
    date_parsed = datetime.strptime(
        date_raw,
        '%Y-%m-%dT%H:%M:%S'
    )
    print(date_parsed)

    url = 'https://nnov.hse.ru/ba/ling/students/news/825211450.html'
    response = requests.get(url, timeout=3)
    main_bs = BeautifulSoup(response.text, 'lxml')
    day_bs = main_bs.find('div', class_='post-meta__day')
    month_bs = main_bs.find('div', class_='post-meta__month')
    year_bs = main_bs.find('div', class_='post-meta__year')
    date_raw = f'{day_bs.text} {month_bs.text} {year_bs.text}'
    print(f'Before: {date_raw}')
    months = {
        'янв': '01',
        # other months are going here... lecturer is lazy
        'апр': '04'
    }
    for month_name, month_number in months.items():
        if month_name in date_raw:
            date_raw = date_raw.replace(
                month_name,
                month_number
            )
    print(date_raw)
    date_parsed = datetime.strptime(
        date_raw,
        '%d %m %Y'
    )
    print(date_parsed)


if __name__ == '__main__':
    main()
