# Lab 02 — Web Scraping, Feature Engineering and EDA — Report

**Course:** Introduction to Artificial Intelligence · **Instructor:** Ali Hassan Sherazi
**Student:** Aman

---

## 1. Task 2.1 — NumPy vs a native Python loop

Summing a 10,000,000-element array:

| Method | Result | Time |
|---|---|---|
| `np.sum(data)` | 4999462.87 | 0.00540 sec |
| Manual `for` loop | 4999462.87 | 1.18568 sec |

**NumPy was ~220x faster** on this machine, for an identical numerical result. The two
implementations agree to the printed precision, so the speed-up is free — nothing about
correctness is traded away. The reason is that `np.sum` runs a single call into compiled C code
operating on a contiguous memory buffer, while the Python `for` loop pays the interpreter's
per-iteration overhead (bytecode dispatch, type checks, boxing/unboxing of each float into a
Python object) ten million times over. Every other library in this course — pandas, spaCy,
scikit-learn — is built on the same principle: push the loop down into compiled code and describe
the computation as operations over whole arrays.

## 2. Task 2.2/2.3 — Scraping and feature engineering

Five articles were scraped from `techncruncher.blogspot.com` with `requests` + `BeautifulSoup`
(`lxml` parser), using the site's Blogger template selectors (`h3.post-title`,
`div.post-body.entry-content`). All 5 of 5 URLs parsed successfully and are saved in
`articles_raw.csv` / `articles.csv` with the full engineered feature set (`num_tokens`,
`num_sentences`, `num_entities`, `num_nouns`, `title_length`, `sentiment_polarity`,
`sentiment_subjectivity`).

**Named-entity spot check** (first 500 tokens of article 1) surfaced dates (`2025`), organisation
names (`Content Ideation and Scriptwriting`, `Brand`), monetary values (`$20`, `$200`) and
cardinal numbers (`10`) — all correctly typed, confirming the NER feature is picking up real
structure rather than noise.

**Sentiment**: all five articles score mildly-to-moderately **positive** (mean polarity ≈ 0.2) and
**moderately-to-highly subjective**, consistent with promotional "best AI tools" / product-review
content — an opinion that the tools are good, not evidence that they are.

## 3. Task 2.4 — Exploratory Data Analysis

- **Title-length and polarity histograms** (`figures/01_*.png`, `figures/02_*.png`): with n = 5,
  these are for visual inspection only, not distributional claims (see Exercise 3).
- **Pair plot** (`figures/03_pairplot.png`) and its correlation matrix show `num_tokens`,
  `num_sentences` and `num_entities` moving together, as expected — longer articles have more
  sentences and more opportunities to mention named entities.
- **TF–IDF top terms** (`figures/04_tfidf_top_terms.png`): with English stop words removed, the
  top-ranked terms are genuinely topical — content/tool/AI/creation/marketing-type words —
  because IDF can no longer be swamped by function words (see Exercise 4).

## 4. In-Lab Exercises

**Exercise 1 (2 more URLs, 7 total).** Added one more Blogger *post* (parsed cleanly) and one
Blogger static *Page* (`/p/privacy-policy.html`, deliberately chosen). Result: **6 of 7 pages
parsed**; the Privacy Policy page failed because Blogger renders **Pages** through a different
template widget than **Posts** — it has no `<h3 class="post-title">` and its body isn't wrapped
in `post-body entry-content`, so both selectors correctly return `None` and the code's existing
"no content block" branch skips it. The fix is either template-specific selectors per URL type, or
filtering non-post URLs out before scraping.

**Exercise 2 (`num_verbs`, `avg_sentence_length`).** Implemented and added to an extended pair
plot (`figures/05_pairplot_extended.png`); both features integrate cleanly with the rest of the
feature set.

**Exercise 3 (KDE over 5 points).** A KDE over 5 points is dominated by sampling noise (standard
error scales as $1/\sqrt n$) and by the kernel's own bandwidth/shape assumption rather than by
evidence between the sparse observed points, so it cannot be reported as "the distribution" of a
population. Instead: show the raw 5 values (rug/strip/swarm plot) labelled explicitly as
descriptive-only, n = 5, and postpone smoothed/",population" language until the sample is
meaningfully larger.

**Exercise 4 (TF–IDF, `stop_words=None`).** Terms appearing only when stop words are kept: `and,
as, be, can, for, in, is, it, of, on, that, the, to, with, you, your` — i.e. exactly the function
words a stop-word list exists to remove. Without removal, raw term frequency lets these
near-universal words dominate every document's top-20 list (they appear dozens of times per
article and IDF can't discount them, since IDF only penalises terms common *across documents*, not
words that happen to be common in English generally), crowding out the actually topical terms.

## 5. Home Assignment — two-source comparison

**Sources.** blog-a = `techncruncher.blogspot.com` (10 articles, AI/marketing-tools blog); blog-b
= `machinelearningmastery.com` (10 articles, applied-ML blog), chosen for topical contrast while
remaining scrapeable with simple `requests` + `BeautifulSoup`. Combined corpus saved as
`two_source_corpus.csv` (20 rows, labelled by `source`).

**Results** (Welch's independent-samples t-test, since the two blogs are unrelated samples with no
reason to assume equal variance):

| Metric | blog-a mean | blog-b mean | difference (a − b) | t | p |
|---|---|---|---|---|---|
| Sentiment polarity | 0.1979 | 0.1102 | **+0.0877** | 4.049 | **0.0011** |
| Noun density (nouns/tokens) | 0.2381 | 0.2265 | +0.0116 | 1.259 | 0.2243 |

Grouped bar chart: `figures/06_home_assignment_grouped_bars.png`.

**Interpretation.** The **polarity** difference is both numerically the larger of the two and
statistically significant at the conventional α = 0.05 threshold (p = 0.0011): the marketing/tools
blog really does read more positively than the technical ML blog, consistent with promotional
"top 10 tools" writing skewing upbeat versus explanatory tutorial writing staying neutral. The
**noun-density** difference is small and not statistically distinguishable from chance at this
sample size (p = 0.22) — both blogs are prose-heavy how-to/explanatory writing, so their
part-of-speech composition is similar, and 10 articles per source gives limited power to detect
anything but a large effect on this metric. In both cases the 10-per-blog samples are convenience
samples (whatever each blog happened to publish), not random draws from "all possible articles" by
each author, so even the significant polarity result describes *these twenty articles* rather than
a guaranteed, permanent property of the two blogs.

## 6. Deliverables checklist

- [x] `Lab_02_Web_Scraping_and_EDA.ipynb` — executed, all outputs visible, no errors
- [x] `articles.csv` — Task 2.2/2.3 corpus with engineered features
- [x] `two_source_corpus.csv` — home-assignment 20-article labelled corpus
- [x] `figures/01–06_*.png` — all six figures
- [x] `report.md` — this file
