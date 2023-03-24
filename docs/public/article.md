# Article package

The `article` package is responsible for handling the articles you have collected from your website.

> **HINT**: In case you think you have found a mistake in this package, contact your assistant. 
> Those who considerably improve this module will get additional bonuses.

The package contains the following modules:
1. [constants.py](#constants)
2. [article.py](#article)
3. [ud.py](#ud)
4. [io.py](#io)

## <a name="constants"></a>constants.py

Module `constants.py` defines an `ArtifactType` class. It represents the types of artifacts 
that can be created by text processing pipelines, such as `CLEANED`, `MORPHOLOGICAL_CONLLU` and `FULL_CONLLU`. 
The description of each artifact you can find in [Dataset requirements](#dataset.md).

> **HINT**: You should utilize attributes of `ArtifactType` in order to save processed versions of files. 
> Otherwise, if you pass a string itself to some saving function, your code will be much more fragile.

## <a name="article"></a>article.py

Module `article.py` represents the methods to work with the `Article` abstraction. 

> **NOTE**: Also there is a useful `split_by_sentence(text)` function which you can use in `lab_6` 
> to split text to list of sentences.

`Article` class is responsible for storing article raw, meta and `conllu` data. 

> **HINT**: Do not forget to create a new instance of the `Article` class to use its wonderful methods)

During the implementation of the labs, you will work with the following methods:

* `set_pos_info(pos_freq_dict)` - use to add POS information in meta file;
* `get_raw_text()` - use to get raw text from the article;
* `set_conllu_sentences(sentences)` - use to set the `conllu_sentences` attribute.

## <a name="ud"></a>ud.py

Module `ud.py` contains functions which are responsible for parsing `CONLL-U`. 

* `parse_conllu_token(token_line)`

It parses the raw text in the `CONLLU` format, for example,
`2\tпроизошло\tпроисходить\tVERB\t_\tGender=Neut|Number=Sing|Tense=Past\t0\troot\t_\t_`, 
into the `CONLL-U` token abstraction using `ConlluToken`, `MorphologicalTokenDTO` 
and `SyntacticTokenDTO`from `lab_6_pipeline/conllu.py` module. 
 
* `extract_sentences_from_raw_conllu(conllu_article_text)`

It extracts sentences from the `CONLL-U` formatted article and stores them in a preferable way. 

> **NOTE**: You do not need to use them, they are used in `from_conllu(path_to_conllu_file)` function in the module `io.py`.


## <a name="io"></a>io.py

Module `io.py` provide functions to work with input/output operations for the `Article` abstraction. 

It consists of the following functions, which are grouped by usage in the labs:

### Lab_5

* `to_raw(article)` - use to save raw texts of each article;
* `to_meta(article)` - use to save meta-information about each article.

### Lab_6

* `from_raw(path_to_raw_data)` - use to load raw texts and create the `Article` abstraction;
* `to_cleaned(article)` - use to save cleaned texts of each article, i.e. lowercased texts with no punctuation;
* `to_meta(article)` - use to save POS information about each article;
* `from_meta(path_to_meta_data)` - use to load meta-information about each article and create the `Article` abstraction;
* `to_conllu(article)` - use to save morphological and syntactic information from the `Article` abstraction into the `conllu` file;
* `from_conllu(path_to_conllu_file)` - use to load morphological and syntactic information and create the `Article` abstraction.
