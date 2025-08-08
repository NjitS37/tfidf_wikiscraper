# -*- coding: utf-8 -*-

########################################################################################
#
# Name: TF-IDF-Wikiscraper
#
# tfidf_wikiscraper.py
#
#########################################################################################

import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import os


def contentscraper(link, language):
    """Scrapes content from a Wikipedia article and removes HTML elements."""
    html = urlopen(link).read()
    soup = BeautifulSoup(html, features="html.parser")

    # Remove all script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    content_area = soup.find(id="mw-content-text")

    if language == "EN":
        # English references header
        references_header = content_area.find("h2", string=re.compile(r"References", re.I))
        if references_header:
            content = []
            for element in content_area.find_all():
                if element == references_header:
                    break
                if element.name in ["p", "ul", "ol"]:
                    content.append(element.text)
            text = "\n".join(content)
        else:
            text = content_area.get_text()

    elif language == "NL":
        # Dutch references headers
        ref_headers = [
            content_area.find("h2", string=re.compile(r"Referenties", re.I)),
            content_area.find("h2", string=re.compile(r"Bronnen, noten en/of referenties", re.I)),
            content_area.find("h2", string=re.compile(r"Noten", re.I))
        ]
        if any(ref_headers):
            content = []
            for element in content_area.find_all():
                if element in ref_headers:
                    break
                if element.name in ["p", "ul", "ol"]:
                    content.append(element.text)
            text = "\n".join(content)
        else:
            text = content_area.get_text()

    # Structure the scraped text
    text = re.sub(r"\[\d+\]", "", text)
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return '\n'.join(chunk for chunk in chunks if chunk)


def linklist(url, language):
    """Extracts all Wikipedia article hyperlinks from a given page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    titles_written = [url]
    allLinks = soup.find(id="bodyContent").find_all("a")

    for link in allLinks:
        if 'href' not in link.attrs:
            continue
        if link['href'].find("/wiki/") == -1 or ":" in link['href']:
            continue

        if language == "EN":
            full_url = "https://en.wikipedia.org" + link['href']
        else:
            full_url = "https://nl.wikipedia.org" + link['href']

        if full_url not in titles_written:
            titles_written.append(full_url)

    return titles_written


def custom_tokenizer(text, language, extra_stopwords):
    """Tokenizes words and filters out common Wikipedia-specific terms."""
    words = re.findall(r'\b[a-zA-Z\'-]{3,}\b', text.lower())

    if language == "EN":
        stopwords = ["citation", "citation needed", "isbn", "issn", "displaystyle", "wikipedia", "Wikipedia", "creative", "commons"]
    else:
        stopwords = ["citaat", "citaat nodig", "isbn", "issn", "displaystyle", "bron", "bewerken", "brontekst", "artikel", "wikipedia", "Wikipedia", "creative", "commons"]

    stopwords.extend(extra_stopwords)
    return [word for word in words if word not in stopwords]


def wikiscraper(input_arg, output_file, N, ngram_min, ngram_max, include_weights, language):
    """Scrapes Wikipedia articles, applies TF-IDF, and outputs the top N words."""

    print(f"Start scraping ({language}).")

    t0 = time.time()

    # Accept either file path or single URL
    if os.path.isfile(input_arg):
        with open(input_arg, 'r', encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        urls = [input_arg.strip()]

    # Load stopwords for Dutch if needed
    extra_stopwords = []
    if language == "NL" and os.path.exists("stopwoorden.txt"):
        with open('stopwoorden.txt', 'r', encoding='utf-8') as f:
            extra_stopwords = [regel.strip().lower() for regel in f]

    stop_words_param = "english" if language == "EN" else extra_stopwords

    vectorizer = TfidfVectorizer(
        ngram_range=(ngram_min, ngram_max),
        stop_words=stop_words_param,
        tokenizer=lambda text: custom_tokenizer(text, language, extra_stopwords),
        token_pattern=None
    )

    all_results = []
    linklengte = 0
    scraped_count = 0

    for url in urls:
        content = []
        links = linklist(url, language)
        linklengte += len(links)

        for i in links:
            content.append(contentscraper(i, language))
            scraped_count += 1

            if scraped_count % 50 == 0:
                print(f"{scraped_count}/{linklengte} links scraped")

        if not content:
            continue

        X = vectorizer.fit_transform(content)
        tfidf_tokens = vectorizer.get_feature_names_out()
        result = pd.DataFrame(data=X.toarray(), columns=tfidf_tokens)

        data = result.T
        data["gemiddelde"] = data.mean(axis=1)
        data = data.sort_values(by="gemiddelde", ascending=False)

        all_results.append(data[["gemiddelde"]].reset_index())

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.columns = ['word', 'score']
        final_df['score'] = (final_df['score'] * 10000).astype(int)
        final_df = final_df.groupby("word").mean().sort_values(by="score", ascending=False)
        top_terms = final_df.head(N).index.tolist()
    else:
        top_terms = []

    with open(output_file, 'w', encoding="utf-8") as f:
        for term in top_terms:
            if include_weights:
                f.write(f"{int(final_df.loc[term, 'score'])} {term}\n")
            else:
                f.write(term + '\n')

    t1 = time.time()
    print(linklengte, "articles have been scraped.")
    print(f"Generating the wordlist took {t1 - t0:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Wikipedia scraper for word ranking (EN/NL)")
    parser.add_argument("--input", required=True, help="Path to input file with URLs or pasted in single URL")
    parser.add_argument("--output", required=True, help="Path to output file")
    parser.add_argument("--N", type=int, default=10000, help="Number of words to output (default: 10000)")
    parser.add_argument("--ngram_min", type=int, default=1, help="Minimum n-gram size (default: 1)")
    parser.add_argument("--ngram_max", type=int, default=1, help="Maximum n-gram size (default: 1)")
    parser.add_argument("--include_weights", action="store_true", help="Include word weights in the output")
    parser.add_argument("--language", choices=["EN", "ENG", "NL"], required=True, help="Language of the Wikipedia scraper")

    args = parser.parse_args()

    language = "EN" if args.language.upper() in ["EN", "ENG"] else "NL"

    wikiscraper(args.input, args.output, args.N, args.ngram_min, args.ngram_max, args.include_weights, language)

if __name__ == "__main__":
    main()

