"""
Crawler implementation
"""
import os
from typing import Pattern, Union
from pathlib import Path
import datetime
import requests
from bs4 import BeautifulSoup
from core_utils.article.article import Article
from core_utils.config_dto import ConfigDTO
import json


class IncorrectSeedURLError(Exception):
    pass


class NumberOfArticlesOutOfRangeError(Exception):
    pass


class IncorrectHeadersError(Exception):
    pass


class IncorrectEncodingError(Exception):
    pass


class IncorrectTimeoutError(Exception):
    pass


class IncorrectVerifyError(Exception):
    pass


class IncorrectHeadlessError(Exception):
    pass


class Config:
    """
    Unpacks and validates configurations
    """

    def __init__(self, path_to_config: Path) -> None:
        """
        Initializes an instance of the Config class
        """
        self.path_to_config = path_to_config
        self.config_data = self._extract_config_content()
        self._validate_config_content()

    def _extract_config_content(self) -> ConfigDTO:
        """
        Returns config values
        """
        with open(self.path_to_config) as file:
            config_data = json.load(file)
        return ConfigDTO(**config_data)

    def _validate_config_content(self) -> None:
        """
        Ensure configuration parameters
        are not corrupt
        """
        if not self.seed_urls:
            raise IncorrectSeedURLError("Seed urls must be provided")
        if not isinstance(self.total_articles_to_find_and_parse, int) or self.total_articles_to_find_and_parse <= 0:
            raise NumberOfArticlesOutOfRangeError("Total articles to find and parse must be a positive integer")
        if not isinstance(self.headers, dict):
            raise IncorrectHeadersError("Headers must be a dictionary")
        if not isinstance(self.encoding, str):
            raise IncorrectEncodingError("Encoding must be a string")
        if not isinstance(self.timeout, int) or self.timeout <= 0:
            raise IncorrectTimeoutError("Timeout must be a positive integer")
        if not isinstance(self.verify_certificate, bool):
            raise IncorrectVerifyError("Verify certificate must be a boolean")
        if not isinstance(self.headless_mode, bool):
            raise IncorrectHeadlessError("Headless mode must be a boolean")

    def get_seed_urls(self) -> list[str]:
        """
        Retrieve seed urls
        """
        return self.config_data.seed_urls

    def get_num_articles(self) -> int:
        """
        Retrieve total number of articles to scrape
        """
        return self.config_data.total_articles_to_find_and_parse

    def get_headers(self) -> dict[str, str]:
        """
        Retrieve headers to use during requesting
        """
        return self.config_data.headers

    def get_encoding(self) -> str:
        """
        Retrieve encoding to use during parsing
        """
        return self.config_data.encoding

    def get_timeout(self) -> int:
        """
        Retrieve number of seconds to wait for response
        """
        return self.config_data.timeout

    def get_verify_certificate(self) -> bool:
        """
        Retrieve whether to verify certificate
        """
        return self.config_data.verify_certificate

    def get_headless_mode(self) -> bool:
        """
        Retrieve whether to use headless mode
        """
        return self.config_data.headless_mode


def make_request(url: str, config: Config) -> requests.models.Response:
    """
    Delivers a response from a request
    with given configuration
    """
    headers = config.get_headers()
    timeout = config.get_timeout()
    verify = config.get_verify_certificate()
    response = requests.get(url, headers=headers, timeout=timeout, verify=verify)
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
        self.config = config
        self.urls = []

    def _extract_url(self, article_bs: BeautifulSoup) -> str:
        """
        Finds and retrieves URL from HTML
        """
        all_links_bs = article_bs.find_all('a')

        for link_bs in all_links_bs:
            href = link_bs.get('href')
            if href is None:
                continue
            elif href.startswith('https://chelny-izvest.ru/news/') and href.count('/') == 5:
                return href

    def find_articles(self) -> None:
        """
        Finds articles
        """
        for seed_url in self.config.get_seed_urls():
            response = make_request(seed_url, self.config)
            if response.status_code != 200:
                continue

            article_bs = BeautifulSoup(response.text, 'lxml')
            article_url = self._extract_url(article_bs)
            if article_url is not None:
                self.urls.append(article_url)

    def get_search_urls(self) -> list:
        """
        Returns seed_urls param
        """
        return self.config.get_seed_urls()


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
        self.article = Article(full_url, article_id)

    def _fill_article_with_text(self, article_soup: BeautifulSoup) -> None:
        """
        Finds text of article
        """
        text_elements = article_soup.find("div", class_="page-main__text").find_all("p")
        text = "\n".join([p.get_text().strip() for p in text_elements])
        self.article.text = text

    def _fill_article_with_meta_information(self, article_soup: BeautifulSoup) -> None:
        """
        Finds meta information of article
        """
        author_elem = article_soup.find('div', class_='page-main__publish-data').find('a', class_='page-main__publish-author global-link')
        authors = [author_elem.text.strip()] if author_elem else ["NOT FOUND"]

        date_elem = article_soup.find('div', class_='page-main__publish-data').find('a', class_='page-main__publish-date')
        date_str = date_elem.get_text(strip=True) if date_elem else None
        date = self.unify_date_format(date_str)

        category_elem = article_soup.find('div', class_='panel-group').find('a', class_='panel-group__title global-link')
        category = category_elem.get_text(strip=True) if category_elem else None

        title_elem = article_soup.find('div', class_='page-main').find('h1', class_='page-main__head')
        title = title_elem.text.strip() if title_elem else None

        self.article.authors = authors
        self.article.date = date
        self.article.category = category
        self.article.title = title

    def unify_date_format(self, date_str: str) -> datetime.datetime:
        """
        Unifies date format
        """
        date_obj = datetime.datetime.strptime(date_str, '%d %B %Y - %H:%M')
        return date_obj

    def parse(self) -> Union[Article, bool, list]:
        """
        Parses each article
        """
        response = requests.get(self.full_url, headers=self.config.headers)
        response.encoding = 'utf-8'
        article_soup = BeautifulSoup(response.text, 'html.parser')
        self._fill_article_with_text(article_soup)
        self._fill_article_with_meta_information(article_soup)

        return self.article


def prepare_environment(base_path: Union[Path, str]) -> None:
    """
    Creates ASSETS_PATH folder if no created and removes existing folder
    """
    assets_path = Path(base_path)
    if os.path.exists(assets_path):
        os.remove(assets_path)
    os.makedirs(assets_path)


def main() -> None:
    """
    Entrypoint for scrapper module
    """
    pass


if __name__ == "__main__":
    main()
