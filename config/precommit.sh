set -x

python -m pylint --exit-zero --rcfile config/stage_1_style_tests/.pylintrc core_utils config seminars

mypy core_utils config seminars

python -m flake8 --config ./config/stage_1_style_tests/.flake8 core_utils config seminars
