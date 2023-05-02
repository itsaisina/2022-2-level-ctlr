set -x

source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

PR_NAME="$1"
PR_AUTHOR="$2"
LAB_PATH="$3"

python config/skip_check.py --pr_name "$PR_NAME" --pr_author "$PR_AUTHOR" --lab_path "$LAB_PATH"

if [ $? -eq 0 ]; then
  echo 'skip check due to special conditions...' && exit 0
fi

TARGET_SCORE=$(bash config/get_mark.sh "$LAB_PATH")

ls
mkdir -p tmp/articles
mv *_raw.txt tmp/articles

if [[ ${TARGET_SCORE} != 4 ]]; then
  mv *_meta.json tmp/articles
fi

if [[ "$LAB_PATH" == "lab_6_pipeline" ]]; then
  mv *_cleaned.txt tmp/articles

  if [[ ${TARGET_SCORE} -gt 4 ]]; then
    mv *_pos_conllu.conllu tmp/articles
    mv *_meta.json tmp/articles
  fi

  if [[ ${TARGET_SCORE} -gt 6 ]]; then
    mv *_morphological_conllu.conllu tmp/articles
  fi
fi

ls -la tmp/articles
