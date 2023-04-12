#!/bin/bash

set -x

echo -e '\n'
echo 'Running pymarkdownlnt check...'

source venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH}"

FAILED=0
LABS=$(cat config/labs.txt)

for LAB_NAME in $LABS; do
 echo "Running pymarkdownlnt for lab ${LAB_NAME}"

 for filename in ${LAB_NAME}/*.md; do
   echo "Running pymarkdownlnt for ${filename}"
   python -m pymarkdown --config config/stage_1_style_tests/pymarkdownlnt.json scan ${filename}

   if [[ $? -ne 0 ]]; then
     echo "pymarkdownlint check failed for ${filename}."
     FAILED=1
   else
     echo "pymarkdownlint check passed for ${filename}."
   fi
  done
done

echo "Running pymarkdownlnt for README.md"
python -m pymarkdown --config config/stage_1_style_tests/pymarkdownlnt.json scan README.md docs/**/*.md

if [[ $? -ne 0 ]]; then
  echo "pymarkdownlnt check failed for common files."
  FAILED=1
else
  echo "pymarkdownlnt check passed for common files."
fi


if [[ ${FAILED} -eq 1 ]]; then
  echo "Pymarkdownlnt check failed."
  exit ${FAILED}
fi

echo "Pymarkdownlnt check passed."

