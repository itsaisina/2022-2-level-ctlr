# Working with the UD text description format: CONLL-U

## Structure

1. One word per line with the following structure:
    1. **ID**: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes (decimal numbers can be lower than 1 but must be greater than 0).

    2. **FORM**: Word form or punctuation symbol.
    
    3. **LEMMA**: Lemma or stem of word form.
    
    4. **UPOS**: Universal part-of-speech tag.
    
    5. **XPOS**: Language-specific part-of-speech tag; underscore if not available.
    
    6. **FEATS**: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
       * As per [this format](https://universaldependencies.org/u/feat/index.html)
       * Structured as `FeatureName=Value|FeatureName=Value|FeatureName=Value...` 
         * where `FeatureName` is a name of the morphological feature of the token (for example, `Number`) and `Value` is the actual value of the feature (for example, `Sing` - short for singular)
    
    7. **HEAD**: Head of the current word, which is either a value of ID or zero (0).
    
    8. **DEPREL**: Universal dependency relation to the HEAD (root if HEAD = 0) or a defined language-specific subtype of one.
       * As per [this format](https://universaldependencies.org/u/dep).
    
    9. **DEPS**: Enhanced dependency graph in the form of a list of head-deprel pairs.
    
    10. **MISC**: Any other annotation.
2. Fields cannot be empty. If no value for a field, the `_` is used.
3. Comments are denoted using `#`. Comments usually consist of the sentences, see the example below.
4. New sentences start with the token ID being `1`.
5. Example:
   ```
   # sent_id = 2003Armeniya.xml_1
   # text = В советский период времени число ИТ- специалистов в Армении составляло около десяти тысяч.
   1	В	в	ADP	_	_	3	case	3:case	_
   2	советский	советский	ADJ	_	Animacy=Inan|Case=Acc|Degree=Pos|Gender=Masc|Number=Sing	3	amod	3:amod	_
   3	период	период	NOUN	_	Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing	11	obl	11:obl:в:acc	_
   4	времени	время	NOUN	_	Animacy=Inan|Case=Gen|Gender=Neut|Number=Sing	3	nmod	3:nmod:gen	_
   5	число	число	NOUN	_	Animacy=Inan|Case=Acc|Gender=Neut|Number=Sing	11	obj	11:obj	_
   6	ИТ	ИТ	PROPN	_	Animacy=Inan|Case=Nom|Gender=Neut|Number=Sing	8	compound	8:compound	SpaceAfter=No
   7	-	-	PUNCT	_	_	6	punct	6:punct	_
   8	специалистов	специалист	NOUN	_	Animacy=Anim|Case=Gen|Gender=Masc|Number=Plur	5	nmod	5:nmod:gen	_
   9	в	в	ADP	_	_	10	case	10:case	_
   10	Армении	Армения	PROPN	_	Animacy=Inan|Case=Loc|Gender=Fem|Number=Sing	5	nmod	5:nmod:в:loc	_
   11	составляло	составлять	VERB	_	Aspect=Imp|Gender=Neut|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act	0	root	0:root	_
   12	около	около	ADP	_	_	14	case	14:case	_
   13	десяти	десять	NUM	_	Case=Gen|NumType=Card	14	nummod	14:nummod	_
   14	тысяч	тысяча	NOUN	_	Animacy=Inan|Case=Gen|Gender=Fem|Number=Plur	11	nsubj	11:nsubj	SpaceAfter=No
   15	.	.	PUNCT	_	_	11	punct	11:punct	_
   ```
6. Line explanation:
   * Line: `2	советский	советский	ADJ	_	Animacy=Inan|Case=Acc|Degree=Pos|Gender=Masc|Number=Sing	3	amod	3:amod	_`
     1. `2` - ID
     2. `советский` - text of the token
     3. `советский` - lemma of the token
     4. `ADJ` - POS
     5. `_` - language specific POS; none in this case
     6. `Animacy=Inan|Case=Acc|Degree=Pos|Gender=Masc|Number=Sing` - morphological features of the token as per [tags](https://universaldependencies.org/u/feat/index.html)
        1. `Animacy=Inan` - inanimate
        2. `Case=Acc` - accusative case
        3. `Degree=Pos` - degree of comparison: positive/first degree
        4. `Gender=Masc` - masculine gender
        5. `Number=Sing` - singular number
     7. `3` - the ID of the HEAD for the current token. HEAD is `период` in this case
     8. `amod` - relation to the HEAD token. `amod` - adjectival modifier as per [tags](https://universaldependencies.org/u/dep/amod.html)
     9. `3:amod` - pair of HEAD:RELATION for the current token
     10. `_` - any other annotation; none in this case

**NB**: file with example text can be found in [here](data/ud_test.conllu). It is a subset of a file from [this repository](https://github.com/UniversalDependencies/UD_Russian-SynTagRus).

## Parsing morphological information from PyMorphy & Mystem and converting it into the UD format

The number of available morphological features in PyMorphy, Mystem and UD is big. It would require a lot of work to make a perfect (i.e., 1-to-1) mapping of the features from one system to another. This section describes how to parse the output from PyMorphy and Mystem and convert **_part of the available morphological features_** to the UD format.

The features selected for conversion:

* Part of speech
* Case
* Number
* Gender
* Animacy
* Tense

The complete mapping of features from PyMorphy and Mystem to the UD format can be found in the [`tags_mapping.json`](data/tags_mapping.json).

To convert from one system to another, it is necessary to understand how each system is structured. 

### Universal Dependency

The UD formatting for morphological tags was partly described in the beginning of the document. 

It is structured as follows:

* `FeatureName=Value|FeatureName=Value|FeatureName=Value...`

Where `FeatureName` is a name of the morphological feature of the token (for example, `Number`) and `Value` is the actual value of the feature (for example, `Sing` - short for singular).

Example:
* `Animacy=Inan|Case=Acc|Degree=Pos|Gender=Masc|Number=Sing`
  1. `Animacy=Inan` - inanimate
  2. `Case=Acc` - accusative case
  3. `Degree=Pos` - degree of comparison: positive/first degree
  4. `Gender=Masc` - masculine gender
  5. `Number=Sing` - singular number

The list of all tags used in the UD system is available on the [dedicated page](https://universaldependencies.org/u/feat/index.html).

### Mystem
The list of all tags used in Mystem is available on the [dedicated documentation page](https://yandex.ru/dev/mystem/doc/grammemes-values.html).

The `pymystem3` library is a wrapper around Mystem for Python. It provides the morphological analysis for the language tokens in the following format:

* `POS,FeatureValue?,FeatureValue?...=(FeatureValue?,FeatureValue?...|FeatureValue?,FeatureValue?...|)`

Where `POS` is a part of speech, `FeatureValue` - the value of a morphological feature, `?` means that the value might not be present, 
`|` means that there are several possible sets of morphological features for the token.

The specific structure varies based on the part of speech, see examples below.

Examples:
* Noun: `S,муж,од=(вин,ед|род,ед)`, `S,жен,од=им,ед`, `S,сокр=(пр,мн|пр,ед|вин,мн|вин,ед|дат,мн|дат,ед)`
  * `S` - Noun
  * `муж` - Masculine gender
  * `од` - animate
  * `вин,ед|род,ед` - `вин,ед` OR `род,ед`
    * `вин,ед` - accusative case, singular number
    * `род,ед` - genitive case, singular number
* Adjective: `A=(вин,ед,полн,муж,од|род,ед,полн,муж|род,ед,полн,сред)`, `A=(вин,мн,полн,неод|им,мн,полн)`
* Verb: `V,нп=прош,мн,изъяв,сов`, `V,пе=(прош,ед,прич,кр,муж,сов,страд|непрош,мн,изъяв,3-л,сов)`
* Numeral-Adjective: `ANUM=(пр,ед,жен|дат,ед,жен|род,ед,жен|твор,ед,жен|)`
* Pronoun-Adjective: `APRO=(пр,мн|дат,мн|род,мн|твор,мн|им,мн|им,ед,жен|вин,ед,муж,од|род,ед,муж|род)`
* Numeral: `NUM=(им|вин,неод)`
* Pronoun: `SPRO,ед,3-л,жен=им`, `SPRO,ед,3-л,жен=(дат|твор)`
* Particle, preposition, interjection, conjunction, part of a compound word, adverb, pronominal adverb: `PART=, PR=, INTJ=, CONJ=, COM=, ADV=, ADVPRO=` respectively

As the `pymystem3` returns these morphological features as a string, there is nothing left but to parse it by delimiters (`,=|()`) <sub><sup>or find more elegant way</sup></sub>.

Additionally, as it is seen from the examples above, the number of features can be different even for the same part of speech: 
compare `S,жен,од=им,ед` and `S,сокр=(пр,мн|пр,ед|вин,мн|вин,ед|дат,мн|дат,ед)`. This fact should be kept in mind while parsing the morphological features.

Parsing of `S,муж,од=(вин,ед|род,ед)` to the UD format:

* `S` - `NOUN`
* `муж` - `Masc`
* `од` - `Anim`
* `вин` - `Acc`
* `ед` - `Sing`
* `род` - `Gen`

**NB**: The complete mapping of features from PyMorphy and Mystem to the UD format can be found in the [`tags_mapping.json`](data/tags_mapping.json).

For the sake of simplicity, only the first possible set of features (i.e., `вин,ед`) is considered.

As per the UD format, the morphological features are structured as following:

* `FeatureName=Value|FeatureName=Value|FeatureName=Value...`

In this case, the resulting string would be the following:

* `Animacy=Anim|Case=Acc|Gender=Masc|Number=Sing`

**NB**: the names of the features in the UD format can be found on the [dedicated page](https://universaldependencies.org/u/feat/index.html).

### PyMorphy
PyMorphy uses the tags from OpenCorpora. The list of all tags is available on the [OpenCorpora Website](http://opencorpora.org/dict.php?act=gram&order=priority).

As the `pymorphy2` returns these morphological features as an instance of the `OpencorporaTag` class, it is possible to access its attributes to extract the information.

Available attributes for `OpencorporaTag` are:
* `POS`
* `animacy`
* `aspect`
* `case`
* `gender`
* `involvement`
* `mood`
* `number`
* `person`
* `tense`
* `transitivity`
* `voice`

They can be accessed as, for example, `tags.animacy` where `tags` is an instance of `OpencorporaTag`.

Parsing of `OpencorporaTag('NOUN,anim,masc sing,nomn')` to the UD format:

* `NOUN` - `NOUN`
* `masc` - `Masc`
* `anim` - `Anim`
* `nomn` - `Nom`
* `sing` - `Sing`

**NB**: The complete mapping of features from PyMorphy and Mystem to the UD format can be found in the [`tags_mapping.json`](data/tags_mapping.json).

As different parts of speech have different tags (for example: _**verbs**_ have _**tense**_ while _**nouns**_ do not), it is important to make the parsing 
per part of speech.

As per the UD format, the morphological features are structured as following:

* `FeatureName=Value|FeatureName=Value|FeatureName=Value...`

In this case, the above `OpencorporaTag` would convert to the following in the UD format:

* `Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing`

**NB**: the names of the features in the UD format can be found on the [dedicated page](https://universaldependencies.org/u/feat/index.html).
