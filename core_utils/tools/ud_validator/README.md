# `core_utils/tools/ud_validate`

`validate.py` is a modified tool. It uses files in `data` directory, taken from 
the [Universal Dependencies tools repository](https://github.com/UniversalDependencies/tools).

It reads a `CoNLL-U` file and verifies that it complies with the UD specification. 

This is an example of running validation:
```
python core_utils/tools/ud_validator/validate.py --lang ru --max-err=0 --level 5 --no-space-after --multiple-roots  --no-tree-text  --no-space-after core_utils/tools/ud_validator/data/test.conllu
```

For proper usage, you can change only **--level 5** parameter

About levels:

**Level 1**: Test only CoNLL-U backbone.

**Level 2**: Test UD format.

**Level 3**: Test UD contents. 

**Level 4**: Test Language-specific labels. 

**Level 5**: Test Language-specific contents.