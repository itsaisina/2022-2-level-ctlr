set -x

source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

python config/config_param_changer.py --config_path="lab_5_scrapper/scrapper_config.json"

echo "Changed config params"

python config/collect_coverage/coverage_analyzer.py
