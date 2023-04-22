set -ex

echo -e '\n'
echo "Check files processing on student dataset"

TARGET_SCORE=$(bash config/get_mark.sh lab_6_pipeline)


if [[ ${TARGET_SCORE} != 0 ]]; then
  mkdir -p tmp/articles
  mv *_cleaned.txt tmp/articles
  mv *_raw.txt tmp/articles

  if [[ ${TARGET_SCORE} -gt 4 ]]; then
    mv *_pos_conllu.conllu tmp/articles
    mv *_meta.json tmp/articles
  fi
  
  if [[ ${TARGET_SCORE} -gt 6 ]]; then
    mv *_morphological_conllu.conllu tmp/articles
  fi
  
  bash config/stage_3_pipeline_tests/s3_5_student_text_preprocess.sh
else
  echo "Skip stage"
fi
