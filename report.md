# Laboratory 3 — Report
*Transforming Textual and Image Data into a Machine-Understandable Format*

Files: `text_features.ipynb` (Part A + Ex. 1–2), `image_features.ipynb` (Part B + Ex. 3–4 + Home Assignment), `image_utils.py`, `dataset.npz`.

**Stated defaults (methodological note).** `CountVectorizer` was used with its defaults: `lowercase=True` (so *The*/*the* merge) and `token_pattern=r"(?u)\b\w\w+\b"` (so single-character tokens such as *"a"* are dropped). Images are converted to RGB, resized to a fixed size (64×64 in the dataset), converted to grayscale (`'L'`) and flattened row-major, guaranteeing a fixed vector length.

---

## Part A — Discussion questions (A1–A3)

**1. What does each column in the resulting matrix represent?**
Each column represents **one unique term (word) from the vocabulary** learned from the whole corpus — i.e. one *feature*. The value in a cell is the **number of times that term occurs in that document**. Here the vocabulary has 11 terms, so every document becomes a point in an 11-dimensional space. Columns are ordered alphabetically (`day, good, great, is, shining, sun, sunny, the, today, weather, wonderful`).

**2. In this specific numerical matrix, what is the machine-understandable format?**
The machine-understandable format is the **document–term matrix**: a $3 \times 11$ matrix of non-negative integers in which each **row is a fixed-length numeric vector** describing one document (e.g. Doc 1 = `[0,0,0,1,1,1,0,1,1,0,0]`). Documents that originally had different lengths and structure are all mapped onto vectors of the same length (11), so algorithms such as Naive Bayes, Logistic Regression or SVM can consume them. Because most entries are 0 (here 18 of 33 cells ≈ 54.5 %, and >99 % for real corpora), scikit-learn stores it as a **sparse matrix**.

**3. How is this format an oversimplification of the original text?**
Information that is **discarded**:
- **Word order and grammar** — "dog bites man" and "man bites dog" get identical vectors.
- **Context and meaning** — *sun* and *sunny* are unrelated columns; synonyms are not linked; negation ("not good") is invisible.
- **Case and punctuation** — everything is lowercased, punctuation removed (*The*/*the* merge).
- **Short tokens** — the single-character word *"a"* is silently dropped by the default `token_pattern`.
- **Relative importance** — raw counts treat frequent function words (*the*, *is*) as equally informative as rare content words (*shining*); a longer document gets larger numbers simply because it is longer.
- **Unseen words** — any word not in the training vocabulary is ignored at prediction time.

---

## Part B — Discussion questions (B1–B4)

**1. How does grayscale conversion act as a simple feature-engineering technique?**
Grayscale conversion maps the three colour channels (R, G, B) of every pixel onto **one luminance value**, $L = 0.299R + 0.587G + 0.114B$. It is feature engineering because it *transforms and reduces* the raw input using domain knowledge: for many tasks (digits, edges, shapes, textures) brightness structure carries the signal and colour is redundant. It cuts the number of features **three-fold** ($H\times W\times 3 \to H\times W$), lowering memory, computation and the risk of over-fitting, at the cost of **all colour information**.

**2. What is the length of the flattened vector after grayscale conversion of an $H \times W$ image?**
$H \times W$ (one intensity value per pixel; the three channels have already been collapsed). For the sample image, $512 \times 512 = 262{,}144$; for a $100\times100$ image, $10{,}000$.

**3. Why is flattening necessary before Logistic Regression or an SVM?**
These models expect a **2-D design matrix of shape (n_samples, n_features)**, in which each sample is **one row vector**. An image is a 2-D (or 3-D) grid, so it must be unrolled (row-major) into a 1-D vector to become one row of that matrix; the model then learns one weight per pixel position. The price is that the **spatial neighbourhood structure is discarded**: the model cannot know that pixel *i* and pixel *i + W* are vertical neighbours. (Convolutional networks avoid this by working on the grid directly — Laboratory 8.)

**4. Why resize all images to a consistent size (e.g. $64\times64$) first?**
- **Fixed vector length:** images of different sizes would give vectors of different lengths that cannot be stacked into one matrix or fed to a model with a fixed number of inputs.
- **Feature alignment:** with equal size, feature *j* always refers to the *same spatial location* in every image, so weights learned for it are comparable across images.
- **Efficiency and control of dimensionality:** a 64×64 image has 4,096 features instead of hundreds of thousands, reducing memory, training time and over-fitting risk.
- Trade-off: fine detail is lost (and aspect ratio distorted if the image is not square).

---

## Home-Lab Exercises

### Exercise 1 — `ngram_range=(1,2)`
**Result and explanation.** The vocabulary grows from **11 to 24 features** (11 unigrams + 13 bigrams).

A bigram is added for **every distinct pair of adjacent tokens** in the corpus. A document with *n* tokens contributes *n − 1* bigrams, so the number of bigram features grows roughly **linearly with corpus size** — and, because most word pairs occur only once, almost every bigram is a brand-new column. On a real corpus with a vocabulary of *V* words, the number of possible bigrams is up to *V²*, so the matrix becomes much wider and much sparser. The benefit is that local word order is partly recovered (*"sunny day"*, *"is good"*); the cost is dimensionality, memory and sparsity.

### Exercise 2 — fourth document with *sun* × 5
**Result and explanation.** Adding `"sun sun sun sun sun"` adds **one new row**; the vocabulary does not change (no new word), and the `sun` column becomes `[1, 1, 0, 5]`. Doc 4 now has the largest value in the whole matrix even though it contains no information except a single repeated word.

**Raw counts favour long documents** because a count is an *absolute* number: the longer a document is, the more often every word is likely to occur, so long documents get larger entries and larger vector norms. A distance- or dot-product-based model (kNN, SVM, cosine before normalisation) then treats "length" as if it were "relevance". The repetition of *sun* five times also makes Doc 4 look five times "more about the sun" than Doc 1, although it just repeats a word. Remedies are **normalising counts** (relative frequency, L1/L2 norm), **TF-IDF** weighting, or **binary** presence (`CountVectorizer(binary=True)`).

### Exercise 3 — 64×64 and 32×32
**Result.** Flattened lengths (after grayscale): original $512\times512$ = **262,144**; $64\times64$ = **4,096**; $32\times32$ = **1,024**.

**What is no longer visible.** At 64×64 the overall pose, face/helmet shape and large light–dark regions survive, but fine detail (fabric texture, small lettering/patches, hair, sharp edges) is gone and edges look blocky. At 32×32 only very coarse blobs remain: the face is barely recognisable, small objects disappear and individual pixels are clearly visible. Each halving of the side length removes 75 % of the pixels; the discarded information is **high-spatial-frequency detail**, and it cannot be recovered by enlarging the image again.

### Exercise 4 — `image_to_features(path, size)`
Implemented in `image_utils.py` (load → RGB → resize → grayscale → flatten). Verified on three images of different sizes — 512×512, 451×300 and 600×400 (W×H) — all returned a vector of length **4,096** at `size=(64, 64)`.

---

## Home Assignment
Dataset: 20 images (10 per class, varying sizes), each processed with `image_to_features(..., size=(64,64))`, stacked into `X` of shape **(20, 4096)** with labels `y` of shape **(20,)**, saved to `dataset.npz` with `numpy.savez`.

**Dimensionality of the feature space:** $64 \times 64 = $ **4,096 dimensions**; the dataset is $X \in \mathbb{R}^{20 \times 4096}$ with $y \in \{0,1\}^{20}$.

**Is twenty samples in 4,096 dimensions a sound basis for learning?** No. With only 20 samples for 4,096 features the ratio is ≈ 0.005 samples per feature, far below what is needed to estimate a model reliably. Because $p \gg n$, the data are always **linearly separable** in such a space, so a linear model (Logistic Regression, SVM) can fit the 20 training points perfectly by exploiting arbitrary pixel-level noise — it will **over-fit** and its training accuracy says nothing about generalisation. This is the **curse of dimensionality**: volume grows exponentially with the number of dimensions, the 20 points are extremely sparse in that space, distances become nearly uniform, and the number of samples required to cover the space grows exponentially. Moreover, each of the 4,096 pixel features is highly redundant with its neighbours, so the *effective* dimensionality of the data is much smaller than 4,096 — which is exactly what **dimensionality reduction (Laboratory 6: PCA, etc.)** exploits by projecting onto a few informative components, and what a **CNN (Laboratory 8)** exploits by sharing a small set of local filters across the image instead of learning one independent weight per pixel. Data augmentation, more samples, strong regularisation and cross-validation are further mitigations.
