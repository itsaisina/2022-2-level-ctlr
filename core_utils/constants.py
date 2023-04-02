"""
Useful constant variables
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_PATH = PROJECT_ROOT / 'tmp' / 'articles'
CRAWLER_CONFIG_PATH = PROJECT_ROOT / 'lab_5_scrapper' / 'scrapper_config.json'

NUM_ARTICLES_UPPER_LIMIT = 150
TIMEOUT_LOWER_LIMIT = 0
TIMEOUT_UPPER_LIMIT = 60
