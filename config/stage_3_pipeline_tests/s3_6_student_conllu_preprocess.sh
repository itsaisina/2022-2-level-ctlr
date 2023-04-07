set -ex

echo "Stage 3B: Student conllu preprocessing"
echo "Starting tests for student dataset"

TARGET_SCORE=$(bash config/get_mark.sh lab_6_pipeline)

source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

if [[ ${TARGET_SCORE} -gt 4 ]]; then
  echo "Running validation"
  for file in tmp/articles/*.conllu; do
    python core_utils/tools/ud_validator/validate.py --lang ru --max-err=0 --level 5 --no-space-after --multiple-roots  --no-tree-text $file
  done
else
  echo "Skip stage"
fi

echo "Conllu files are checked. Done"
