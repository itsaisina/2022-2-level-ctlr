# Article package

The `article` package is responsible for handling the articles you have collected from your website.

> **HINT**: In case you think you have found a mistake in this package, contact your assistant. 
> Those who considerably improve this module will get additional bonuses.

The package contains the following modules:
1. [`article`](#article)
2. [`ud`](#ud)
3. [`io`](#io)

## <a name="article"></a>`article` module

`article` module represents the methods to work with the `Article` abstraction.

`Article` class is responsible for storing article raw, meta and `conllu` data and working with it.
During the implementation of Lab 5 and Lab 6, you should use the methods of this class, 
so we advise you to study them.

> **HINT**: Do not forget to create a new instance of the `Article` class to use its methods.

In addition to the `Article` class, the module has:

1. `date_from_meta(date_txt)` function which converts text date to `datetime` object.
2. `get_article_id_from_filepath(path_to_file)` function which extracts the article id from its path.
3. `split_by_sentence(text)` function which you can use to split text to list of sentences in Lab 6.
4. `SentenceProtocol` class which you should inherit for `ConlluSentence` class in Lab 6.
5. `ArtifactType` class which provides types of artifacts that can be created by text processing pipelines,
such as `CLEANED`, `MORPHOLOGICAL_CONLLU`, `POS_CONLLU` and `FULL_CONLLU`. 
   1. The description of each artifact you can find in [Dataset requirements](#dataset.md).

> **HINT**: You should utilize attributes of `ArtifactType` in order to 
> save processed versions of files. 
> Otherwise, if you pass a string itself to some saving function, 
> your code will be much more fragile.

## <a name="ud"></a>`ud` module

`ud` module contains functions which are responsible for parsing `CONLL-U`.
 
`OpencorporaTagProtocol` class represents the abstraction definition for `pymorphy2.tagset.OpencorporaTag`.
When you implement Lab 6 for mark 10, you should use `pymorphy2` for morphological analyzes.
`pymorphy2` analyzing each token get you the result of analysis 
as the `OpencorporaTagProtocol` token instance that you need to pass 
as a parameter to `OpenCorporaTagConverter` class methods.


`extract_sentences_from_raw_conllu(conllu_article_text)` function extracts sentences from 
the `CONLL-U` formatted article and stores them in a preferable way: 

```python
    [
        {
            'position': sentence_position,
            'text': sentence_text,
            'tokens': sentence_tokens
        },
        {
            'position': sentence_position,
            'text': sentence_text,
            'tokens': sentence_tokens
        },
        ...
    ]
```

> **HINT**: This function will be useful when implementing the Lab 6.

`TagConverter` class is responsible for tags conversion between different formats, 
in your case from Mystem to UD and from PyMorphy to UD (for mark 10).
You need to inherit its interface and implement the following abstraction inside the 
`pipeline.py` file.

> **NOTE**: To use it, you should have an information about 
> the correspondence of one format tags to another format tags. 
> For more details refer to the [Lab 6 guide](../../lab_6_pipeline/README.md).

`TagConverter` class stores information about POS and morphological tags and contains two methods - 
`convert_morphological_tags(tags)` and `convert_pos(tags)` - that you need to implement for Lab 6.

## <a name="io"></a>`io` module

`io` module provide functions to work with input/output operations for the 
`Article` abstraction. 

It consists of the following functions, which are grouped by usage in the labs:

### Lab_5

* `to_raw(article)` - use to save raw texts of each article;
* `to_meta(article)` - use to save meta-information about each article.

### Lab_6

* `from_raw(path_to_raw_data, article)` - use to load raw texts and create the
   `Article` abstraction;
* `to_cleaned(article)` - use to save cleaned texts of each article, i.e. 
   lowercased texts with no punctuation;
* `to_meta(article)` - use to save POS information about each article;
* `from_meta(path_to_meta_data, article)` - use to load meta-information about 
   each article and create the `Article` abstraction;
* `to_conllu(article, include_morphological_tags: bool, pymorphy: bool)` - 
   use to save morphological and syntactic information from the `Article` abstraction 
   into the `conllu` file;
