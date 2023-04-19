# pylint: disable=too-few-public-methods, disable=too-many-arguments
"""
ConfigDTO class implementation: stores the configuration information
"""


class ConfigDTO:
    """
    Type annotations for configurations
    """

    seed_urls: list[str]
    total_articles: int
    headers: dict[str, str]
    encoding: str
    timeout: int
    should_verify_certificate: bool
    headless_mode: bool

    def __init__(self,
                 seed_urls: list[str],
                 total_articles_to_find_and_parse: int,
                 headers: dict[str, str],
                 encoding: str,
                 timeout: int,
                 should_verify_certificate: bool,
                 headless_mode: bool
                 ):
        """
        Initializes an instance of the ConfigDTO class
        """

        self.seed_urls = seed_urls
        self.total_articles = total_articles_to_find_and_parse
        self.headers = headers
        self.encoding = encoding
        self.timeout = timeout
        self.should_verify_certificate = should_verify_certificate
        self.headless_mode = headless_mode
