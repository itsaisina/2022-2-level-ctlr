set -ex

echo -e '\n'
echo "Check files processing on student dataset"

TARGET_SCORE_SCRAPPER=$(bash config/get_mark.sh lab_5_scrapper)
TARGET_SCORE_PIPELINE=$(bash config/get_mark.sh lab_6_pipeline)


if [[ ${TARGET_SCORE_SCRAPPER} != 0 || ${TARGET_SCORE_PIPELINE} != 0 ]]; then
  mkdir -p tmp/articles
  mv *_raw.txt tmp/articles
  mv *_cleaned.txt tmp/articles || echo "no files to move"

  if [[ ${TARGET_SCORE_SCRAPPER} -gt 4 || ${TARGET_SCORE_PIPELINE} -gt 4 ]]; then
    mv *_meta.json tmp/articles
    mv *_pos_conllu.conllu tmp/articles || echo "no files to move"
  fi
  if [[ ${TARGET_SCORE_PIPELINE} -gt 6 ]]; then
    mv *_morphological_conllu.conllu tmp/articles || echo "no files to move"
  fi
else
  echo "Skip unpacking dataset"
fi
