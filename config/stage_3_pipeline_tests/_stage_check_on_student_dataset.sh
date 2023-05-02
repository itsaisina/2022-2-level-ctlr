set -ex

echo -e '\n'
echo "Check processing on student dataset"

TARGET_SCORE=$(bash config/get_mark.sh lab_6_pipeline)

source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

if [[ ${TARGET_SCORE} != 0 ]]; then
  bash config/unpack_archived_dataset.sh
  python lab_6_pipeline/pipeline.py
  ls -la tmp/articles
else
  echo "Skip stage"
fi
