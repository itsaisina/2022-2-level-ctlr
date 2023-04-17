set -x

source venv/bin/activate

export PYTHONPATH=$(pwd):$PYTHONPATH
python config/skip_check.py --pr_name "$1" --pr_author "$2" --lab_path "lab_5_scrapper"
if [ $? -eq 0 ]; then
  echo 'skip check due to special conditions...' && exit 0
fi

python config/config_param_changer.py --config_path="lab_5_scrapper/scrapper_config.json"

echo "Changed config params"

python lab_5_scrapper/scrapper.py

echo "Collected dataset"

echo "Checking volume of files"

TARGET_SCORE=$(bash config/get_mark.sh lab_5_scrapper)

if [[ ${TARGET_SCORE} == 4 ]]; then
  echo "Running score four checks"
  python -m pytest -m "mark4 and stage_2_4_dataset_volume_check" --capture=no --ignore=lab_6_pipeline
else
  python -m pytest -m "mark10 and stage_2_4_dataset_volume_check" --capture=no --ignore=lab_6_pipeline
fi

ret=$?
if [ "$ret" = 5 ]; then
  echo "No tests collected.  Exiting with 0 (instead of 5)."
  exit 0
fi

echo "Pytest results (should be 0): $ret"

exit "$ret"

ls -la tmp/articles
