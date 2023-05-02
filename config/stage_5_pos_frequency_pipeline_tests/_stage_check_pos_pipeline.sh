set -ex

echo -e '\n'
echo "Check files processing on student dataset"

TARGET_SCORE=$(bash config/get_mark.sh lab_6_pipeline)

source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

if [[ ${TARGET_SCORE} != 0 ]]; then
  bash config/unpack_archived_dataset.sh

  if [[ ${TARGET_SCORE} == 10 ]]; then
    python lab_6_pipeline/pos_frequency_pipeline.py
  fi

  echo "POSFrequencyPipeline is checked. Done"
  echo "Your solution is accepted! Proceed to further tasks from your lecturer."
else
  echo "Skip stage"
fi
