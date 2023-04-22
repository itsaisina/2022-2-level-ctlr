# Dataset requirements

For effective analysis of the collected articles, it is necessary to organize the data in a
consistent way.
The description of the structure and each of the elements of the dataset is provided below.

## Structure

```text
+-- 2022-2-level-ctlr
    +-- tmp
        +-- articles
            +-- articles
                +-- 1_raw.txt <- the raw text of the article with the ID as the name
                +-- 1_cleaned.txt <- lowercased text with no punctuation
                +-- 1_pos_conllu.conllu <- UD text format with POS tags
                +-- 1_morphological_conllu.conllu <- UD text format with morphological tags
                +-- 1_meta.json <- the meta-information of the article
                +-- 2_raw.txt
                +-- 2_cleaned.txt
                +-- 2_pos_conllu.conllu <- UD text format with POS tags
                +-- 2_morphological_conllu.conllu
                +-- 2_meta.json
                +-- ...
                +-- 100_raw.txt
                +-- 100_cleaned.txt
                +-- 100_pos_conllu.conllu <- UD text format with POS tags
                +-- 100_morphological_conllu.conllu
                +-- 100_meta.json
```

## Raw texts

Raw articles texts are stored in `X_raw.txt` files where `X` corresponds to the index
of the article. The text is not preprocessed in any way.

Example:

```text
Жители Китая, у которых дома живут три кошки, сняли на видео развлечение, которое придумали
питомицы. При этом главным реквизитом игры стала, как ни странно, рисоварка. Хотя на первый
взгляд сложно представить, что же могло заинтересовать животных в обычном кухонном агрегате.
Оказывается, пузырьки, которые то и дело возникают на крышке рисоварки. Эти пузырьки крайне
занимательно прихлопывать лапами.
```

## Processed texts

Ideally, the dataset consists of three processed texts examples: cleaned text, text with
morphology annotation and text with syntactic annotation.

### Cleaned text

Cleaned texts are stored in `X_cleaned.txt` files where `X` corresponds to the index
of the article.

Cleaned text is lowercase and does not include any punctuation.
Word forms are the same as in the raw text.

Example:

```text
жители китая у которых дома живут три кошки сняли на видео развлечение которое придумали питомицы
при этом главным реквизитом игры стала как ни странно рисоварка хотя на первый взгляд сложно
представить что же могло заинтересовать животных в обычном кухонном агрегате оказывается пузырьки
которые то и дело возникают на крышке рисоварки эти пузырьки крайне занимательно прихлопывать лапами
```

### POS annotation

Texts with POS annotation are stored in `X_pos_conllu.conllu` files where `X` corresponds to
the index of the article.

The files contain the following information about the tags: `ID`, `FORM`, `LEMMA`, `UPOS`, `XPOS`.

Read more about the structure of such files in [UD format description](ud_format.md).

Example for the first sentence of the sample article:

```text
# sent_id = 0
# text = Жители Китая, у которых дома живут три кошки, сняли на видео развлечение,
которое придумали питомицы.
1   Жители  житель  NOUN    _   _   0   root    _   _
2   Китая   китай   NOUN    _   _   0   root    _   _
3   у   у   ADP _   _   0   root    _   _
4   которых который ADJ _   _   0   root    _   _
5   дома    дома    ADV _   _   0   root    _   _
6   живут   жить    VERB    _   _   0   root    _   _
7   три три NUM _   _   0   root    _   _
8   кошки   кошка   NOUN    _   _   0   root    _   _
9   сняли   снимать VERB    _   _   0   root    _   _
10  на  на  ADP _   _   0   root    _   _
11  видео   видео   NOUN    _   _   0   root    _   _
12  развлечение развлечение NOUN    _   _   0   root    _   _
13  которое который ADJ _   _   0   root    _   _
14  придумали   придумывать VERB    _   _   0   root    _   _
15  питомицы    питомица    NOUN    _   _   0   root    _   _
16  .   .   PUNCT   _   _   0   root    _   _
```

### Morphological annotation

Texts with morphological annotation are stored in `X_morphological_conllu.conllu` files where `X`
corresponds to the index of the article.

The files contain the following information about the tags: `ID`, `FORM`, `LEMMA`, `UPOS`, `XPOS`, `FEATS`.

Read more about the structure of such files in [UD format description](ud_format.md).

Example for the first sentence of the sample article:

```text
# sent_id = 0
# text = Жители Китая, у которых дома живут три кошки, сняли на видео развлечение,
которое придумали питомицы.
1   Жители  житель  NOUN    _   Animacy=Anim|Case=Nom|Gender=Masc|Number=Plur   0   root    _   _
2   Китая   китай   NOUN    _   Animacy=Inan|Case=Gen|Gender=Masc|Number=Sing   0   root    _   _
3   у   у   ADP _   _   0   root    _   _
4   которых который ADJ _   Case=Ins|Number=Plur    0   root    _   _
5   дома    дома    ADV _   _   0   root    _   _
6   живут   жить    VERB    _   Number=Plur 0   root    _   _
7   три три NUM _   Case=Nom    0   root    _   _
8   кошки   кошка   NOUN    _   Animacy=Anim|Case=Gen|Gender=Fem|Number=Sing    0   root    _   _
9   сняли   снимать VERB    _   Number=Plur|Tense=Past  0   root    _   _
10  на  на  ADP _   _   0   root    _   _
11  видео   видео   NOUN    _   Animacy=Inan|Case=Ins|Gender=Neut|Number=Plur   0   root    _   _
12  развлечение развлечение NOUN   _  Animacy=Inan|Case=Acc|Gender=Neut|Number=Sing  0  root   _  _
13  которое который ADJ _   Case=Acc|Gender=Neut|Number=Sing    0   root    _   _
14  придумали   придумывать VERB    _   Number=Plur|Tense=Past  0   root    _   _
15  питомицы    питомица    NOUN   _  Animacy=Anim|Case=Gen|Gender=Fem|Number=Sing   0  root   _  _
16  .   .   PUNCT   _   _   15  punct   _   _
```

## Meta information

Meta information is stored in files with `X_meta.json` names where `X` corresponds to
the index of the article.

It is also required to store meta-information. It includes:

1. Article id (a positive integer, it must match the id of the file)
2. Article title (a string)
3. Article date (a string)
4. Article URL  (a string)
5. Article topics (a list of strings)
6. Article author (a list of strings)

Example:

```json
{
    "id": 2,
    "url": "https://www.nn.ru/text/style/2023/03/11/72125285/",
    "title": "«Вы актер или батюшка?» Простой рабочий одевается как Пушкин и ходит так на оборонный завод",
    "date": "2023-03-11 17:30:00",
    "author": [
        "Дарья Манохина"
    ],
    "topics": [
        "Стиль и красота"
    ],
    "pos_frequencies": {}
}
```

## Volume

Aim at collecting not less than `100` articles from your chosen web source.
