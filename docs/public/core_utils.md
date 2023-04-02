# Core utils

Core utils package contains auxiliary materials to help you implement your laboratory works. 

We are going to over each of its parts to guide you when and how to use it. 

Core utils contents:

* [article package](#article-package)
* [tools package](#tools-package)
* [`config_DTO` module](#configurations-dto)
* [`constants` module](#module-with-constants)
* [`visualizer` module](#visualizer-module)
* [tests package](#tests-package)

## Article package

The `article` package is responsible for handling the articles you have collected from your website.
You are going to use it for both laboratory №5 and laboratory №6.

Exhaustive guide is located [here](article.md).

## Tools package

This package contains [`ud_validator`](../../core_utils/tools/ud_validator) module. 
It is responsible for verifying that the generated CoNLL-U files match the format. 
For more details on the CoNLL-U format, refer to [UD format description](ud_format.md).

As a part of your laboratory №6, you are to perform a morphological analysis of the collected articles.
Results must be saved as a CoNLL-U file. The correctness of the file format can be verified via 
[script prepared for you](../../core_utils/tools/ud_validator/validate.py). To learn how to use it,
refer to the [instruction](../../core_utils/tools/ud_validator/README.md).

## Configurations DTO

The `config_dto` module defines a `ConfigDTO` abstraction. 
This abstraction is responsible for indicating what fields must be passed 
as a configuration settings along with what their types must be. 

According to this abstraction, configuration parameters are as follows:

* `seed_urls`: a list of strings
* `total_articles_to_find_and_parse`: an integer
* `headers`: a dictionary with string keys and string values
* `encoding`: a string
* `timeout`: an integer
* `verify_certificate`: a boolean value
* `headless_mode`: a boolean value

They match the fields of the [`scrapper_config.json`](../../lab_5_scrapper/scrapper_config.json) configuration file.
For more details on what each of the parameters presents, refer to 
[the laboratory 5 guide](../../lab_5_scrapper/README.md#configuring-scrapper).

> **NOTE**: During implementation of laboratory №5, make sure to return a `ConfigDTO` instance from the 
`Config._extract_config_content` method. 

## Module with constants

[`constance.py`](../../core_utils/constants.py) module defines the following constant values:

* `PROJECT_ROOT`: a path to [`2022-2-level-ctlr`](../..) folder, which the root of the current project
* `ASSETS_PATH`: a path to [`2022-2-level-ctlr/tmp/article`](../../tmp/articles) folder, where all the collected articles must be stored
* `CRAWLER_CONFIG_PATH`: a path to the [`scrapper_config.json`](../../lab_5_scrapper/scrapper_config.json) file with configuration parameters for scrapper
* `NUM_ARTICLES_UPPER_LIMIT`: a maximum number for articles to be collected, anything above this number must be considered invalid
* `TIMEOUT_LOWER_LIMIT`: a minimum number of seconds for a timeout, anything below this number must be considered invalid
* `TIMEOUT_UPPER_LIMIT`: a maximum number of seconds for a timeout, anything above this number must be considered invalid

> **NOTE**: Can you tell why the folder for articles is located in the directory with the name `tmp`?

> **NOTE**: Make sure to import these constants from the `constant.py` module and use them whenever you need
> to specify a path or boundary values (for example, when validating configuration values)


## Visualizer module 

As a final part of laboratory №6, students are expected to perform an analysis of distribution of part-of-speech
tags in the processed collected articles. This is where [`visualizer.py` module](../../core_utils/visualizer.py)
comes into play. Its `visuzalize` function takes an `Article` instance along with a path and creates a graph depicting 
POS distribution in the specified location.

Before calling it, make sure you have already filled the `pos_frequencies` field of the corresponding meta file
during [`POSFrequencyPipeline`](../../lab_6_pipeline/pos_frequency_pipeline.py) execution.

> **NOTE**: `visualize` function must be called during the execution of `POSFrequencyPipeline.run` method

> **NOTE**: the name of the resulting image must have the same id as the article analysed.

## Tests package

To make sure that the provided materials work as intended, they are thoroughly tested.
The [tests](../../core_utils/tests) package contains a number of unit-tests for 
[`article` package](../../core_utils/tests/article_test.py),
[`config_dto` module](../../core_utils/tests/config_dto_test.py), and 
[`visualizer`](../../core_utils/tests/visualizer_test.py) module.

During work on laboratory №5 and laboratory №6, you do not need to interact with these tests.
However, should you suspect that the provided modules behave unexpectedly, examination of these tests
may help catch the bug. Any suggestion on improvements of core utils is encouraged and rewarded.  



