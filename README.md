# Technical Track of Computer Tools for Linguistic Research (2021/2022)

As a part of a compulsory course 
[Computer Tools for Linguistic Research](https://www.hse.ru/edu/courses/749661034)
in [National Research University Higher School of Economics](https://www.hse.ru/).

This technical track is aimed at building basic skills for retrieving data from external
WWW resources and processing it for future linguistic research. The idea is to automatically 
obtain a dataset that has a certain structure and appropriate content, 
perform morphological analysis using various natural language processing (NLP) 
libraries. [Dataset requirements](./docs/public/dataset.md).

Instructors: 

* [Khomenko Anna Yurievna](https://www.hse.ru/org/persons/65858472) - linguistic track lecturer
* [Lyashevskaya Olga Nikolaevna](https://www.hse.ru/staff/olesar) - linguistic track lecturer
* [Demidovskij Alexander Vladimirovich](https://www.hse.ru/staff/demidovs#sci) - technical track lecturer
* [Uraev Dmitry Yurievich](https://www.hse.ru/org/persons/208529395) - technical track practice lecturer
* [Kashchikhin Andrei Nikolaevich](https://t.me/WhiteJaeger) - technical track expert
* [Kazyulina Marina Sergeevna](https://t.me/poemgranate) - technical track assistant
* [Zharikov Egor Igorevich](https://t.me/godb0i) - technical track assistant
* [Novikova Irina Alekseevna](https://t.me/iriinnnaaaaa) - technical track assistant
* [Blyudova Vasilisa Mikhailovna](https://t.me/Vasilisa282) - technical track assistant
* [Zaytseva Vita Vyacheslavovna](https://t.me/v_ttec) - technical track assistant

## Project Timeline

1. **Scrapper**
   1. Short summary: Your code can automatically parse a media website you are going to choose, 
      save texts and its metadata in a proper format
   1. Deadline: *TBD*
   1. Format: each student works in their own PR
   1. Dataset volume: 5-7 articles
   1. Design document: [`./lab_5_scrapper/README.md`](./lab_5_scrapper/README.md)
   1. List of media websites to select from: at the `Resources` section on this page
1. **Pipeline**
   1. Short summary: Your code can automatically process raw texts from previous step,
      make point-of-speech tagging and basic morphological analysis.
   1. Deadline: *TBD*
   1. Format: each student works in their own PR
   1. Dataset volume: 5-7 articles
   1. Design document: **TBD**

## Lectures history

|    Date    | Lecture topic                                   | Important links              |
|:----------:|:------------------------------------------------|:-----------------------------|
| 13.03.2022 | **Lecture:** Introduction to technical track.   | [Lab no. 5 description][7]   |
| 17.03.2022 | **Seminar:** 3rd party libraries.               | N/A                          |
| 20.03.2022 | **Lecture:** Requests and `HTML`.               | [Listing][8]                 |
| 24.03.2022 | **Seminar:** Headers and introduction to `bs4`. | [Listing][12]                |
| 03.04.2022 | **Lecture:** Access file system via `pathlib`.  | [Listing][13], [Listing][14] |
| 07.04.2022 | **Seminar:** Early version of `HTMLParser`.     | [Listing][15]                |
| 10.04.2022 | **Lecture:** Working with dates via `datetime`. | [Listing][16]                |

You can find a more complete summary from lectures as a 
[list of topics](./docs/public/lectures_content.md). 
Более полное содержание пройденных занятий в виде списка ключевых тем.

## Technical solution

| Module                | Description                        | Component          | Need to get |
|:----------------------|:-----------------------------------|:-------------------|:------------|
| [`pathlib`][1]        | working with file paths            | scrapper           | 4           |
| [`requests`][2]       | downloading web pages              | scrapper           | 4           |
| [`BeautifulSoup4`][3] | finding information on web pages   | scrapper           | 4           |
| [`lxml`][4]           | **Optional** parsing HTML          | scrapper           | 6           |
|  `datetime`           | working with dates                 | scrapper           | 6           |
|  `json`               | working with json text format      | scrapper, pipeline | 4           |
| [`pymystem3`][5]      | module for morphological analysis  | pipeline           | 6           |
| [`pymorphy2`][6]      | module for morphological analysis  | pipeline           | 10          |

Software solution is built on top of three components:
1. [`scrapper.py`](./lab_5_scrapper/scrapper.py) - a module for finding articles 
   from the given media, extracting text and
   dumping it to the file system. Students need to implement it.
1. `pipeline.py` - a module for processing text: point-of-speech tagging and 
   basic morphological analysis. Students need to implement it.
1. [`article.py`](core_utils/article/article.py) - a module for article abstraction 
   to encapsulate low-level
   manipulations with the article
   
## Handing over your work

Order of handing over:

1. lab work is accepted for oral presentation.
2. a student has explained the work of the program and showed it in action.
3. a student has completed the min-task from a mentor that requires some slight code modifications.
4. a student receives a mark:
   1. that corresponds to the expected one, if all the steps above are completed and mentor is 
      satisfied with the answer;
   2. one point bigger than the expected one, if all the steps above are completed and 
      mentor is very satisfied with the answer;
   3. one point smaller than the expected one, if a lab is handed over one week later than the 
      deadline and criteria from 4.1 are satisfied;
   4. two points smaller than the expected one, if a lab is handed over more than one week later 
      than the deadline and criteria from 4.1 are satisfied.

> NOTE: a student might improve their mark for the lab, if they complete 
> tasks of the next level after handing over
> the lab.

A lab work is accepted for oral presentation if all the criteria below are satisfied:

1. there is a Pull Request (PR) with a correctly formatted name:
   `Scrapper, <NAME> <SURNAME> - <UNIVERSITY GROUP NAME>`. 
   Example: `Scrapper, Valeriya Kuznetsova - 19FPL1`.
2. has a filled file `target_score.txt` with an expected mark. 
   Acceptable values: 4, 6, 8, 10.
3. has green status.
4. has a label `done`, set by mentor.
 
## Resources

1. Academic performance: [link][9]
1. Media websites list: [link][10]
1. Python programming course from previous semester: [link][11]
1. Scrapping tutorials: [YouTube series (russian)](https://youtu.be/7hn1_t2ZtJQ)
1. [HOWTO: Set up your fork](./docs/public/starting_guide.md)
1. [HOWTO: Running tests](./docs/public/tests.md)
1. [HOWTO: Running assignments in terminal](./docs/public/run_in_terminal.md)

[1]: https://pypi.org/project/pathlib/
[2]: https://pypi.org/project/requests/2.25.1/
[3]: https://pypi.org/project/beautifulsoup4/4.11.1/
[4]: https://pypi.org/project/lxml/
[5]: https://pypi.org/project/pymystem3/
[6]: https://pypi.org/project/pymorphy2/
[7]: ./lab_5_scrapper/README.md
[8]: ./seminars/seminar_03_20_2023/try_requests.py
[9]: https://docs.google.com/spreadsheets/d/19DS6F6_NrgjGbLUjFm9-REuuaECvApEW_o4pHvaXyLQ
[10]: https://docs.google.com/spreadsheets/d/11mmZCKW0WK7rZlpg3eOBA074zwWiXgJjivVUIdDe6-E
[11]: https://github.com/fipl-hse/2022-2-level-labs
[12]: ./seminars/seminar_03_24_2023/try_beautiful_soup.py
[13]: ./seminars/seminar_04_03_2023/try_json.py
[14]: ./seminars/seminar_04_03_2023/try_fs.py
[15]: ./seminars/seminar_04_07_2023/try_html_parser.py
[16]: ./seminars/seminar_04_10_2023/try_dates.py
