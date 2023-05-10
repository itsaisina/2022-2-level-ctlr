"""
Crawler implementation
"""
import datetime
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Pattern, Union
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from core_utils.article.article import Article
from core_utils.article.io import to_meta, to_raw
from core_utils.config_dto import ConfigDTO
from core_utils.constants import (ASSETS_PATH, CRAWLER_CONFIG_PATH,
                                  NUM_ARTICLES_UPPER_LIMIT,
                                  TIMEOUT_LOWER_LIMIT, TIMEOUT_UPPER_LIMIT)


class IncorrectSeedURLError(Exception):
    """
    Exception raised when seed_urls value in configuration
    file is not a list of strings or a string is not a valid URL
    """


class NumberOfArticlesOutOfRangeError(Exception):
    """
    Exception raised when total_articles_to_find_and_parse value
    in configuration file is out of range
    """


class IncorrectNumberOfArticlesError(Exception):
    """
    Exception raised when total_articles_to_find_and_parse
    value in configuration file is not an integer greater than 0
    """


class IncorrectHeadersError(Exception):
    """
    Exception raised when headers value in configuration file is not a dictionary
    """


class IncorrectEncodingError(Exception):
    """
    Exception raised when encoding value in configuration file is not a string
    """


class IncorrectTimeoutError(Exception):
    """
    Exception raised when timeout value in configuration file
    is not an integer between 1 and 30
    """


class IncorrectVerifyError(Exception):
    """
    Exception raised when should_verify_certificate
    value in configuration file is not a boolean
    """


class Config:
    """
    Unpacks and validates configurations
    """

    def __init__(self, path_to_config: Path) -> None:
        """
        Initializes an instance of the Config class
        """
        self.path_to_config = path_to_config
        self._validate_config_content()
        config_file = self._extract_config_content()
        self._seed_urls = config_file.seed_urls
        self._num_articles = config_file.total_articles
        self._headers = config_file.headers
        self._encoding = config_file.encoding
        self._timeout = config_file.timeout
        self._should_verify_certificate = config_file.should_verify_certificate
        self._headless_mode = config_file.headless_mode

    def _extract_config_content(self) -> ConfigDTO:
        """
        Returns config values
        """
        with open(self.path_to_config, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return ConfigDTO(**config)

    def _validate_config_content(self) -> None:
        """
        Ensure configuration parameters
        are not corrupt
        """
        config = self._extract_config_content()

        if not isinstance(config.seed_urls, list):
            raise IncorrectSeedURLError("Invalid value for seed_urls in configuration file")

        for seed_url in config.seed_urls:
            if not re.match(r'^https?://.*', seed_url):
                raise IncorrectSeedURLError("Invalid seed URL in configuration file")

        total_articles_to_find_and_parse = config.total_articles
        if not isinstance(total_articles_to_find_and_parse, int) \
                or total_articles_to_find_and_parse < 1:
            raise IncorrectNumberOfArticlesError(
                "Invalid value for total_articles_to_find_and_parse in configuration file")

        if total_articles_to_find_and_parse > NUM_ARTICLES_UPPER_LIMIT:
            raise NumberOfArticlesOutOfRangeError(
                "Invalid value for total_articles_to_find_and_parse in configuration file")

        if not isinstance(config.headers, dict):
            raise IncorrectHeadersError("Invalid value for headers in configuration file")

        if not isinstance(config.encoding, str):
            raise IncorrectEncodingError("Invalid value for encoding in configuration file")

        if not isinstance(config.timeout, int) \
                or config.timeout < TIMEOUT_LOWER_LIMIT or config.timeout > TIMEOUT_UPPER_LIMIT:
            raise IncorrectTimeoutError("Invalid value for timeout in configuration file")

        if not isinstance(config.should_verify_certificate, bool):
            raise IncorrectVerifyError(
                "Invalid value for should_verify_certificate in configuration file")

        if not isinstance(config.headless_mode, bool):
            raise IncorrectVerifyError("Invalid value for headless_mode in configuration file")

    def get_seed_urls(self) -> list[str]:
        """
        Retrieve seed urls
        """
        return self._seed_urls

    def get_num_articles(self) -> int:
        """
        Retrieve total number of articles to scrape
        """
        return self._num_articles

    def get_headers(self) -> dict[str, str]:
        """
        Retrieve headers to use during requesting
        """
        return self._headers

    def get_encoding(self) -> str:
        """
        Retrieve encoding to use during parsing
        """
        return self._encoding

    def get_timeout(self) -> int:
        """
        Retrieve number of seconds to wait for response
        """
        return self._timeout

    def get_verify_certificate(self) -> bool:
        """
        Retrieve whether to verify certificate
        """
        return self._should_verify_certificate

    def get_headless_mode(self) -> bool:
        """
        Retrieve whether to use headless mode
        """
        return self._headless_mode


def make_request(url: str, config: Config) -> requests.models.Response:
    """
    Delivers a response from a request
    with given configuration
    """
    response = requests.get(url,
                            headers=config.get_headers(),
                            timeout=config.get_timeout(),
                            verify=config.get_verify_certificate())
    response.encoding = config.get_encoding()
    time.sleep(random.uniform(1, 2))
    return response


class Crawler:
    """
    Crawler implementation
    """

    url_pattern: Union[Pattern, str]

    def __init__(self, config: Config) -> None:
        """
        Initializes an instance of the Crawler class
        """
        self._seed_urls = config.get_seed_urls()
        self._config = config
        self.urls = []

    def _extract_url(self, article_bs: BeautifulSoup) -> str:
        """
        Finds and retrieves URL from HTML
        """
        href = article_bs.get('href')
        if not isinstance(href, str):
            return ''
        parsed_url = urlparse(href)
        if isinstance(href, str) \
                and parsed_url.scheme == 'https' \
                and parsed_url.netloc == 'chelny-izvest.ru' \
                and parsed_url.path.startswith('/news/') \
                and parsed_url.path.count('/') == 3:
            return href
        return ''

    def find_articles(self) -> None:
        """
        Finds articles
        """
        for seed_url in self._seed_urls:
            response = make_request(seed_url, self._config)
            if response.status_code != 200:
                continue

            article_bs = BeautifulSoup(response.text, 'lxml')
            for elem in article_bs.find_all('a', class_='widget-view-small__head'):
                if len(self.urls) >= self._config.get_num_articles():
                    return

                article_url = self._extract_url(elem)
                if not article_url or article_url in self.urls:
                    continue

                self.urls.append(article_url)

    def get_search_urls(self) -> list:
        """
        Returns seed_urls param
        """
        return self._seed_urls


class HTMLParser:
    """
    ArticleParser implementation
    """

    def __init__(self, full_url: str, article_id: int, config: Config) -> None:
        """
        Initializes an instance of the HTMLParser class
        """
        self.full_url = full_url
        self.article_id = article_id
        self.config = config
        self.article = Article(self.full_url, self.article_id)

    def _fill_article_with_text(self, article_soup: BeautifulSoup) -> None:
        """
        Finds text of article
        """
        text_elements = article_soup.select("div.page-main__text p")
        self.article.text = "\n".join([p.get_text(strip=True) for p in text_elements])

    def _fill_article_with_meta_information(self, article_soup: BeautifulSoup) -> None:
        """
        Finds meta information of article
        """
        author_elem = article_soup.find('a', {'class': 'page-main__publish-author'})
        self.article.author = [elem.get_text(strip=True) for elem in author_elem] \
            if author_elem else ["NOT FOUND"]

        date_elem = article_soup.find('a', {'class': 'page-main__publish-date'})
        date_str = date_elem.get_text(strip=True)
        self.article.date = self.unify_date_format(date_str)

        topic_elem = article_soup.find_all('a', {'class': 'panel-group__title'})[1]
        self.article.topics = [topic_elem.get_text(strip=True)]

        title_elem = article_soup.find('h1', {'class': 'page-main__head'})
        self.article.title = title_elem.get_text(strip=True)

    def unify_date_format(self, date_str: str) -> datetime.datetime:
        """
        Unifies date format
        """
        months_dict = {
            "января": "January",
            "февраля": "February",
            "марта": "March",
            "апреля": "April",
            "мая": "May",
            "июня": "June",
            "июля": "July",
            "августа": "August",
            "сентября": "September",
            "октября": "October",
            "ноября": "November",
            "декабря": "December"
        }
        date_str = date_str.replace(date_str.split()[1], months_dict[date_str.split()[1]])
        return datetime.datetime.strptime(date_str, '%d %B %Y - %H:%M')

    def parse(self) -> Union[Article, bool, list]:
        """
        Parses each article
        """
        response = make_request(self.full_url, self.config)
        article_soup = BeautifulSoup(response.text, 'lxml')
        self._fill_article_with_text(article_soup)
        self._fill_article_with_meta_information(article_soup)
        return self.article


def prepare_environment(base_path: Union[Path, str]) -> None:
    """
    Creates ASSETS_PATH folder if no created and removes existing folder
    """
    if base_path.exists():
        shutil.rmtree(base_path)
    base_path.mkdir(parents=True)


class CrawlerRecursive(Crawler):
    """
    Recursive Crawler implementation
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.start_url = config.get_seed_urls()[0]
        self.num_visited_urls = 0
        self.last_file_index = 1
        self.visited_urls = []
        self.urls = []
        self.load_crawler_data()

    def load_crawler_data(self) -> None:
        """
        Loads start_url and collected urls
        from a json file into crawler
        """
        crawler_data_path = Path(__file__).parent / 'crawler_data.json'
        if crawler_data_path.exists():
            with open(crawler_data_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    self.last_file_index = data['last_file_idx']
                    self.num_visited_urls = data['num_visited_urls']
                    self.start_url = data['start_url']
                    self.urls = data['urls']
                    self.visited_urls = data['visited_urls']
                except json.JSONDecodeError:
                    return

    def save_crawler_data(self) -> None:
        """
        Saves start_url and collected urls
        from crawler into a json file
        """
        data = {
            'last_file_idx': self.last_file_index,
            'num_visited_urls': self.num_visited_urls,
            'start_url': self.start_url,
            'urls': self.urls,
            'visited_urls': self.visited_urls
        }
        crawler_data_path = Path(__file__).parent / 'crawler_data.json'
        with open(crawler_data_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=True, indent=4, separators=(', ', ': '))

    def find_articles(self) -> None:
        """
        Recursive collecting and searching for links on the site
        """
        if self.num_visited_urls:
            self.start_url = self.visited_urls[self.num_visited_urls - 1]

        response = make_request(self.start_url, self._config)
        links_bs = BeautifulSoup(response.content, 'lxml')

        for link in links_bs.find_all('a'):
            url = self._extract_url(link)
            if url and url not in self.urls and len(self.urls) < self._config.get_num_articles():
                self.urls.append(url)
            elif not url:
                href = link.get("href")
                if href and href not in self.visited_urls \
                        and urlparse(href).netloc == urlparse(self.start_url).netloc:
                    self.visited_urls.append(href)

        self.save_crawler_data()

        if len(self.urls) < self._config.get_num_articles():
            self.num_visited_urls += 1
            self.last_file_index += 1
            self.save_crawler_data()
            self.find_articles()


def main() -> None:
    """
    Entrypoint for scrapper module
    """
    config = Config(path_to_config=CRAWLER_CONFIG_PATH)
    prepare_environment(ASSETS_PATH)
    crawler = Crawler(config=config)
    crawler.find_articles()

    for ind, url in enumerate(crawler.urls, start=1):
        parser = HTMLParser(full_url=url,
                            article_id=ind,
                            config=config)
        article = parser.parse()
        if isinstance(article, Article):
            to_raw(article)
            to_meta(article)


def main_recursive() -> None:
    """
    Entrypoint for scrapper module
    """
    config = Config(path_to_config=CRAWLER_CONFIG_PATH)
    prepare_environment(ASSETS_PATH)
    recursive_crawler = CrawlerRecursive(config=config)
    recursive_crawler.find_articles()

    for index, current_url in enumerate(recursive_crawler.urls[
                                        recursive_crawler.last_file_index - 1:],
                                        start=recursive_crawler.last_file_index):
        parser = HTMLParser(full_url=current_url,
                            article_id=index,
                            config=config)
        parsed_article = parser.parse()
        if isinstance(parsed_article, Article):
            to_raw(parsed_article)
            to_meta(parsed_article)


if __name__ == "__main__":
    main()
