set -ex

echo "Stage 2B: Reference text preprocessing"
echo "Starting tests for admin dataset"

TARGET_SCORE=$(bash config/get_mark.sh lab_6_pipeline)

source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

if [[ ${TARGET_SCORE} == 4 ]]; then
  echo "Running score four checks"
  python -m pytest -m "mark4 and stage_3_4_admin_data_processing" --capture=no
elif [[ ${TARGET_SCORE} == 6 ]]; then
  echo "Running score six checks"
  python -m pytest -m "mark6 and stage_3_4_admin_data_processing" --capture=no
elif [[ ${TARGET_SCORE} == 8 ]]; then
  echo "Running score eight checks"
  python -m pytest -m "mark8 and stage_3_4_admin_data_processing" --capture=no
  echo "TBD: later"
else
  echo "Running score ten checks"
  python -m pytest -m "mark8 and stage_3_4_admin_data_processing" --capture=no
  python -m pytest -m "mark10 and stage_3_6_advanced_morphological_processing" --capture=no

  echo "TODO: check for PosFrequencyPipeline"
  echo "TBD: later"
fi

echo "Raw data is checked. Done"
