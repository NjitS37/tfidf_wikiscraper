# TF-IDF-Wikiscraper
This command-line Python tool scrapes English or Dutch Wikipedia articles for content relevant to a specific topic. It uses either a single starting article URL, or a list of main article URLs provided in a `.txt` file. This code needs an active internet connection to access Wikipedia.
The script collects all hyperlinks within the main articles, then analyzes the content of those articles to extract the most relevant terms. 
The output consists of ranked terms based on the TF-IDF (Term Frequency-Inverse Document Frequency) across all articles. The length of the wordlist and a range of `n-grams` can be specified.

The following Python packages are used in this code:
- `argparse` (for running the code from command line)
- `requests`
- `beautifulsoup4`
- `pandas`
- `urllib.request`
- `re` (Regular Expressions, part of Python's standard library)
- `sklearn` (for TF-IDF analysis)
- `time` (for time-related functionality, part of Python's standard library)

The necessary packages (those not included in Python's standard library), can be installed using `requirements.txt`

Functions of the code:
- Article scraping: The code scrapes the main Wikipedia articles provided in the input `.txt` file or provided single URL.  
- Hyperlink extraction: For each main article, the script extracts all hyperlinks pointing to other relevant Wikipedia articles.  
- TF-IDF analysis: The script computes the Term Frequency-Inverse Document Frequency (TF-IDF) for words found across all the linked articles.  
- Output: The code outputs the top N terms with the highest average TF-IDF scores.

Running the Code:
To run the script, ensure that the `.txt` file with the list of Wikipedia article URLs is prepared with the URLs being on separate lines. A single link can also be pasted after `--input`. Then, execute the Python script.

Example Command:
```bash
python tfidf_wikiscraper.py --input input.txt --output wordlist.txt --N 50000 --ngram_max 2
```

Where `input.txt` is the .txt file containing the list of URLs and `wordlist.txt` is the output file containing the scraped wordlist. `N` is the length of the returned wordlist, and `ngram_min` is the minimum and `ngram_max` is the maximum of how many words a term in the wordlist can exist of. When not specified, the standard value for `N` is 10000, and the standard values for `ngram_min` and `ngram_max` are 1; single words are used. An option `include_weights` is also included, where weights are added before the word separated by a space, so that PCFG merging can be applied to this wordlist. Language can be specified with the option `language`, which can take arguments ENG or EN for English, and NL for Dutch articles.

The script will output the terms with the highest average TF-IDF values for all scraped articles. These terms can be considered the most relevant across the provided Wikipedia articles.

The code is available in English and Dutch. For the Dutch version, a text file with common stop words is provided, which is needed to clean the scraped articles (https://github.com/stopwords-iso/stopwords-nl).
