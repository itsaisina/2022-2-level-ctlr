set -ex

echo "ConlluToken validation"
echo "Starting tests for ConlluToken"

TARGET_SCORE=$(bash config/get_mark.sh lab_6_pipeline)

source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

if [[ ${TARGET_SCORE} == 4 ]]; then
  echo "Running score four checks"
  python -m pytest -m "mark4 and stage_3_3_conllu_token_checks" --capture=no --ignore lab_6_pipeline/tests/s4_pos_frequency_pipeline_test.py --ignore lab_6_pipeline/tests/s3_6_advanced_pipeline_test.py
elif [[ ${TARGET_SCORE} == 6 ]]; then
  echo "Running score six checks"
  python -m pytest -m "mark6 and stage_3_3_conllu_token_checks" --capture=no --ignore lab_6_pipeline/tests/s4_pos_frequency_pipeline_test.py --ignore lab_6_pipeline/tests/s3_6_advanced_pipeline_test.py
elif [[ ${TARGET_SCORE} == 8 ]]; then
  echo "Running score eight checks"
  python -m pytest -m "mark8 and stage_3_3_conllu_token_checks" --capture=no --ignore lab_6_pipeline/tests/s4_pos_frequency_pipeline_test.py --ignore lab_6_pipeline/tests/s3_6_advanced_pipeline_test.py
else
  echo "Running score ten checks"
  python -m pytest -m "mark10 and stage_3_3_conllu_token_checks" --capture=no
fi
echo "ConlluToken is checked. Done"
