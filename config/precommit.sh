set -x

python -m pylint --exit-zero --rcfile config/stage_1_style_tests/.pylintrc lab_5_scrapper core_utils config --ignore lab_5_scrapper/tests

mypy lab_5_scrapper core_utils config

python -m flake8 --config ./config/stage_1_style_tests/.flake8 lab_5_scrapper core_utils config
