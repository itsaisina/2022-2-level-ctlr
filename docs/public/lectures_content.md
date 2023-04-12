# Short summary of lectures

## Lecture 1. Introduction to technical track

Web scraping as a craft. Place of technical track in overall discipline: conceptually and
assessment formula.
Technical track overview. Programming assignment overview. Client-server architecture in 
World Wide Web.
Types of `HTTP` methods: `GET`, `POST`, `DELETE`, `PUT`. Request. Response.

## Seminar 1. 3rd party libraries

Python package manager `pip`. `requirements.txt` as a manifest of project dependencies.
Library `requests` for sending
requests to server. `requests.get` to get `html` code of a page.

## Lecture 2. Requests and `HTML`

Making requests with `requests` API. Idea of mimicking to human-made requests. 
Tip no. 1: random timeouts among calls.
Tip no. 2: sending requests with headers from browser. Obtaining headers. 
API for sending a request with headers.
Check for request status: implicit cast to `bool`, check for status code, 
switch on exception raising. Encoding
overwriting before response processing. Introduction to HTML scraping. 
Key strategies for finding elements: 
by `id`, by class, by tag name, by child-parent relations, and by combination 
of aforementioned approaches.

## Seminar 2. Headers and introduction to `bs4`

Making requests with `requests` API. Extracting headers from browser. Making randomized sleeps 
in code. `bs4`: installation, basic API. Finding elements in `HTML` page with `find`, `find_all`.

## Lecture 3. Access file system via `pathlib`

JSON as a format for storing configuration. Module `json` to read and write to a 
file in `JSON` format. Basic API: `dump` and `load`. Advanced parameters: indentation and 
`ASCII` symbols. Paths as unique description of file position. Paths: relative and absolute.
`pathlib` as the recommended library for creating, writing and reading files. Construction of
paths with forward slash. Recommendation to build paths based on `__file__` global variable.
Basic API: `exists`, `glob`, `mkdir`. Removal of directories: `rmdir` versus `shutil.rmtree`.
Removal of files: `unlink`.

## Lecture 4. Working with dates via `datetime`

Date management as one of most challenging tasks in data analysis. 
Module `datetime` to process dates. Basic API: `datetime` class, static methods
`strftime` for formatting dates and `strptime` for parsing them from string. 
Special directives to construct pattern string to parse. Necessity in additional processing of 
raw date before parsing due to some `locale` requirements.
