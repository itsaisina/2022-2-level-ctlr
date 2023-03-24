# Short summary of lectures

## Lecture 1. Introduction to technical track

Web scraping as a craft. Place of technical track in overall discipline: conceptually and assessment formula.
Technical track overview. Programming assignment overview. Client-server architecture in World Wide Web. 
Types of `HTTP` methods: `GET`, `POST`, `DELETE`, `PUT`. Request. Response.

## Seminar 1. 3rd party libraries

Python package manager `pip`. `requirements.txt` as a manifest of project dependencies. Library `requests` for sending
requests to server. `requests.get` to get `html` code of a page.

## Lecture 2. Requests and `HTML`

Making requests with `requests` API. Idea of mimicking to human-made requests. Tip no. 1: random timeouts among calls.
Tip no. 2: sending requests with headers from browser. Obtaining headers. API for sending a request with headers.
Check for request status: implicit cast to `bool`, check for status code, switch on exception raising. Encoding
overwriting before response processing. Introduction to HTML scraping. Key strategies for finding elements: 
by `id`, by class, by tag name, by child-parent relations, and by combination of aforementioned approaches.
