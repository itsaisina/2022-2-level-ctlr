"""
Utils for lab_5_scrapper tests
"""
import random

from config.test_params import TEST_PATH
from core_utils.article import article
from core_utils.article.io import to_meta, to_raw
from core_utils.constants import ASSETS_PATH, CRAWLER_CONFIG_PATH
from core_utils.tests.utils import copy_student_data
from lab_5_scrapper.scrapper import Config, Crawler, HTMLParser


def scrapper_setup() -> None:
    """
    Set up TEST_PATH for scrapper tests
    """
    if any(ASSETS_PATH.iterdir()):
        copy_student_data()
    else:
        config = Config(CRAWLER_CONFIG_PATH)

        TEST_PATH.mkdir(exist_ok=True)
        article.ASSETS_PATH = TEST_PATH

        crawler = Crawler(config)
        crawler.find_articles()
        parser = HTMLParser(random.choice(crawler.urls), 1, config)
        return_value = parser.parse()
        to_raw(return_value)
        to_meta(return_value)
