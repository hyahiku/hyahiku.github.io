# Data Science: A Structured Curriculum

*Compiled from your coursework: MIS 761 (Quantitative Methods), DA 621 / ITE 621 & ITE 622 (Programming for Data Analytics I & II), ITE 451/651 (Managing Big Data & Web Databases), and MIS 769 (Big Data Analytics for Business).*

---

## How This Curriculum Is Organized

Seven units, thirty-two modules, in dependency order — math feeds statistics, statistics feeds machine learning, programming feeds everything. Each module answers four questions:

- **What it is** — the concept, defined plainly
- **Why & when** — the problem it solves and the situations that call for it
- **How it's applied** — tools, code patterns, and the assignments where you used it
- **The math** — the formulas and intuition behind them, where applicable

**Prerequisite map:** Unit I → Unit II → Unit IV; Unit III (programming) runs in parallel and supports everything; Units V–VII build on Unit IV.

---

# UNIT I — Mathematical Foundations

## Module 1: Calculus for Data Science

**What it is.** The mathematics of change (derivatives) and accumulation (integrals). Includes exponential and logarithmic functions, which model growth and compounding.

**Why & when.** Nearly every model you'll train is fit by *optimization* — finding parameter values that minimize an error function. Optimization is calculus: the minimum of a function sits where its derivative equals zero, and algorithms like gradient descent follow the derivative "downhill." Exponentials/logs appear in compound growth, logistic regression, and information theory.

**How it's applied.**
- Symbolic differentiation and integration with `sympy` (your MIS 761 Assignment 1: `sympy.diff`, `sympy.integrate`)
- Compound interest: discrete vs. continuous compounding — you showed that `n` (compounding periods) drops out under continuous compounding
- Conceptually inside every `.fit()` call: scikit-learn and Keras minimize loss functions via calculus-based solvers

**The math.**
- Derivative (instantaneous rate of change): f′(x) = lim(h→0) [f(x+h) − f(x)] / h
- Power rule: d/dx(xⁿ) = n·xⁿ⁻¹ — so f(x) = x³ gives f′(x) = 3x². Constants vanish: f(x) = x³ − 5 has the *same* derivative (shifting a graph vertically doesn't change its slope — your Assignment 1 Q8)
- Indefinite integral (antiderivative): ∫(x³ + 5)dx = x⁴/4 + 5x + C. Unlike the derivative, adding a constant *does* change the integral, because integration accumulates the function's height (your Q10)
- Optimization: set f′(x) = 0 to find critical points; second derivative tells you min vs. max
- Compound interest: A = P(1 + r/n)^(nt); continuous: A = Pe^(rt)
- Partial derivatives & the gradient: for f(w₁, w₂, …), the gradient ∇f is the vector of partial derivatives — the direction of steepest ascent. Gradient descent updates parameters w ← w − α·∇L(w), where α is the learning rate. This is how neural networks and logistic regression learn.

**Free resource:** [Essence of Calculus — 3Blue1Brown (video series)](https://www.3blue1brown.com/topics/calculus) — visual, intuition-first derivatives and integrals

---

## Module 2: Linear Algebra

**What it is.** The mathematics of vectors (ordered lists of numbers), matrices (grids of numbers), and linear transformations between them.

**Why & when.** A dataset *is* a matrix — rows are observations, columns are features. Regression, PCA, embeddings, and neural networks are all matrix operations. Solving a regression is literally solving a system of linear equations.

**How it's applied.**
- `numpy` arrays: matrix creation, multiplication (`@` / `np.dot`), transpose (`.T`), inverse (`np.linalg.inv`), solving systems (`np.linalg.solve`) — your MIS 761 Assignment 5
- Graphical solutions: plotting two lines and checking intersection — intersecting lines have a solution; parallel lines don't (your A5 Q2–Q3)
- Under the hood of OLS regression, PCA, and Word2Vec similarity

**The math.**
- System of equations: two lines y = (−3/4)x + 5/2 and y = (5/2)x − 3 intersect at exactly one point → unique solution. Parallel lines (same slope, different intercept) → no solution. Same line → infinite solutions
- Matrix form: Ax = b, solved by x = A⁻¹b when A is invertible (det(A) ≠ 0)
- Matrix multiplication: (AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ — dimensions must chain: (m×n)(n×p) → (m×p). Not commutative: AB ≠ BA
- Dot product: a·b = Σaᵢbᵢ = |a||b|cos θ — measures alignment of two vectors; the basis of cosine similarity used later in NLP (Module 30)
- Identity matrix I: AI = A; Inverse: AA⁻¹ = I
- Eigenvalues/eigenvectors: Av = λv — directions v that a transformation only stretches (by λ). The backbone of PCA (Module 24)

**Free resource:** [Essence of Linear Algebra — 3Blue1Brown (video series)](https://www.3blue1brown.com/topics/linear-algebra) — the classic visual treatment of vectors, matrices, and eigenvectors

---

## Module 3: Probability

**What it is.** The mathematics of uncertainty — assigning numbers in [0, 1] to events and reasoning about how events combine and condition on each other.

**Why & when.** Data is generated by random processes; probability is the language for modeling them. Classification models output probabilities; hypothesis tests are probability statements; Naive Bayes is Bayes' theorem applied directly.

**How it's applied.**
- Hand computation of marginal, joint, and conditional probabilities (your MIS 761 Assignment 2 — airline scenarios)
- Choosing the right probability type for a scenario: one event → marginal; two independent events together → joint; "given that" → conditional
- Foundation for Naive Bayes (Module 17), logistic regression (Module 15), and all inference (Module 6)

**The math.**
- Marginal probability: P(A) = favorable / total (1 upgraded plane of 4 → P = 0.25)
- Joint probability (independent events): P(A ∩ B) = P(A)·P(B) (late flight 0.25 × on-time meeting 0.80 = 0.20)
- Conditional probability: P(A|B) = P(A ∩ B) / P(B)
- Independence: A, B independent ⇔ P(A|B) = P(A)
- Complement: P(Aᶜ) = 1 − P(A); Addition rule: P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
- Bayes' theorem: P(A|B) = P(B|A)·P(A) / P(B) — inverts a conditional; "update belief in A after seeing evidence B"
- Random variables & distributions:
  - Binomial: k successes in n independent trials, P(X=k) = C(n,k)pᵏ(1−p)ⁿ⁻ᵏ; symmetric when p = 0.5 (mean = median — your A3 Q3 observation)
  - Normal: bell curve, f(x) = (1/σ√2π)e^(−(x−μ)²/2σ²); defined entirely by mean μ and std σ; ~68/95/99.7% within 1/2/3 σ
- Expected value: E[X] = Σ xᵢP(xᵢ) — the long-run average

**Free resource:** [Probability Library — Khan Academy (course)](https://www.khanacademy.org/math/statistics-probability/probability-library) — worked problems on marginal, joint, and conditional probability

---

# UNIT II — Statistics

## Module 4: Descriptive Statistics & Exploratory Data Analysis (EDA)

**What it is.** Summarizing a dataset's center, spread, and shape numerically and visually before modeling anything.

**Why & when.** Always first. EDA reveals skew, outliers, missing data, and relationships that determine which models and transformations are appropriate. Your Titanic EDA (MIS 761 A3) is the template: describe → visualize → compare groups → correlate.

**How it's applied.**
- `df.describe()`, `df.info()`, `.mean()`, `.median()`, `.value_counts()`
- Histograms and boxplots to see shape (Titanic ages: right skew — mean pulled above median by older passengers)
- Group comparisons: survivors vs. non-survivors age distributions
- Correlation: Age vs. Fare gave r ≈ 0.096 → weak relationship (your A3 Q8)

**The math.**
- Mean: x̄ = Σxᵢ/n — sensitive to outliers. Median: middle value — robust. Mode: most frequent
- Skew diagnosis: mean > median → right (positive) skew; mean < median → left skew; mean ≈ median → symmetric
- Variance: s² = Σ(xᵢ − x̄)²/(n−1); Standard deviation: s = √s² (same units as the data)
- Quartiles & IQR: IQR = Q3 − Q1; a common outlier rule: outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR] (what boxplot whiskers show)
- Covariance: cov(X,Y) = Σ(xᵢ−x̄)(yᵢ−ȳ)/(n−1) — direction of co-movement, unit-dependent
- Pearson correlation: r = cov(X,Y)/(sₓsᵧ) ∈ [−1, 1] — unit-free strength of *linear* association. Rough guide: |r| < 0.3 weak, 0.3–0.7 moderate, > 0.7 strong. Correlation ≠ causation

**Free resource:** [Summarizing Quantitative Data — Khan Academy (course)](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data) — mean/median, spread, boxplots, and skew with practice exercises

---

## Module 5: Sampling & Statistical Inference

**What it is.** Using a sample to draw conclusions about a population, with quantified uncertainty.

**Why & when.** You almost never observe the full population. Inference tells you how far your sample statistic is likely to be from the population truth — the prerequisite for hypothesis testing and for trusting model coefficients.

**How it's applied.**
- Recognizing sampling designs: simple random, stratified, cluster, convenience (MIS 761 A4)
- The Titanic dataset itself is a sample of the passenger population — every test you ran treats it that way
- `train_test_split` (Module 12's ML workflow) is random sampling applied to model validation

**The math.**
- Population parameters (μ, σ) vs. sample statistics (x̄, s)
- Law of Large Numbers: x̄ → μ as n grows
- Central Limit Theorem: the distribution of sample means approaches Normal(μ, σ/√n) regardless of the population's shape (for large n) — why t-tests work even on skewed data
- Standard error: SE = s/√n — the std of the sampling distribution; shrinks with √n
- Confidence interval (95%): x̄ ± t*·SE — "if we repeated the sampling many times, 95% of such intervals would contain μ"

**Free resource:** [Sampling Distributions — Khan Academy (course)](https://www.khanacademy.org/math/statistics-probability/sampling-distributions-library) — CLT and standard error, built up with simulations

---

## Module 6: Hypothesis Testing

**What it is.** A formal procedure for deciding whether an observed effect is real or plausibly just sampling noise.

**Why & when.** Whenever you claim a difference or relationship — "survivors were younger," "class distribution differs from industry standard" — you need to rule out chance. Also underlies every p-value you read in regression output.

**How it's applied (your MIS 761 A4, all with `scipy.stats`):**
- One-sample t-test: is mean passenger age different from a hypothesized 30?
- Two-sample t-test: do survivors and non-survivors differ in average age?
- ANOVA: does average age differ across the 3 passenger classes?
- Chi-square: does the observed class distribution match an expected (industry-standard) distribution?

**The math.**
- Framework: state H₀ (no effect) and H₁; choose α (typically 0.05); compute test statistic; get p-value = P(data this extreme | H₀ true); reject H₀ if p < α
- Errors: Type I = false positive (probability α); Type II = false negative (probability β); power = 1 − β
- One-sample t: t = (x̄ − μ₀)/(s/√n), df = n − 1
- Two-sample t: t = (x̄₁ − x̄₂)/√(s₁²/n₁ + s₂²/n₂)
- ANOVA F-test: F = (between-group variance)/(within-group variance) = MSB/MSW. Large F → group means differ. (ANOVA generalizes the t-test to 3+ groups; testing pairs repeatedly would inflate Type I error)
- Chi-square goodness of fit: χ² = Σ(Oᵢ − Eᵢ)²/Eᵢ over categories — compares observed vs. expected counts. Also used for independence between two categorical variables
- Interpretation discipline: "fail to reject H₀" ≠ "H₀ is true"; statistical significance ≠ practical significance

**Free resource:** [Hypothesis Testing and the Null Hypothesis — StatQuest (video)](https://www.youtube.com/watch?v=0oc49DyA3hU) — the clearest 14-minute intro to H₀, rejection, and p-value logic

---

# UNIT III — Programming & Data Management

## Module 7: Python Foundations

**What it is.** The core language: data types, control flow, functions, and the scientific stack (NumPy, pandas) built on top.

**Why & when.** Python is the lingua franca of data science — every later module runs on it. Fluency in the basics (DA 621's full arc: variables → operators → conditionals → loops → strings/lists/tuples → file I/O → exceptions → searching/sorting → comprehensions → functions/dictionaries) is what lets you spend your attention on the analysis instead of the syntax.

**How it's applied.**
- Data types: `int`, `float`, `str`, `bool`, `list` (mutable, ordered), `tuple` (immutable), `dict` (key→value), `set` (unique items)
- Control flow: `if/elif/else`, `for`, `while`; comprehension shorthand: `[x**2 for x in nums if x > 0]`
- Functions: `def`, parameters/returns, scope; exceptions: `try/except` for graceful failure; file I/O: `open()`, context managers (`with`)
- NumPy: n-dimensional arrays, vectorized math (orders of magnitude faster than loops), broadcasting
- pandas: `DataFrame`/`Series`, `read_csv`, indexing with `.loc`/`.iloc`, filtering with boolean masks, `groupby().agg()`, `merge`, `sort_values` — your workhorse in every assignment
- Regular expressions (`re`, ITE 622 Week 2): pattern matching for text cleaning — `\d+` digits, `\w+` word chars, `re.findall`, `re.sub`

**The math.** Minimal — but algorithmic thinking matters: searching (linear O(n) vs. binary O(log n) on sorted data) and sorting (built-in Timsort, O(n log n)) introduce complexity notation, which explains why vectorized NumPy beats Python loops and why big data needs distributed systems (Unit VI).

**Free resource:** [The Official Python Tutorial (article series)](https://docs.python.org/3/tutorial/) — canonical, free, covers everything in the DA 621 arc; pair with pandas' "10 Minutes to pandas"

---

## Module 8: SQL & Relational Databases

**What it is.** The relational model — data in tables with defined schemas, linked by keys — and SQL, the declarative language for querying it.

**Why & when.** Organizational data lives in relational databases. Before pandas ever sees the data, SQL selects, filters, joins, and aggregates it at the source. Your ITE 451/651 course covers the full stack: basic → conditional → join → "for-all" queries, plus connecting Python to MySQL.

**How it's applied.**
- Schema design: tables, rows, columns; primary key (unique row ID), foreign key (reference to another table's PK) — this enforces referential integrity and enables normalization (storing each fact once)
- Core queries: `SELECT cols FROM t WHERE cond GROUP BY col HAVING cond ORDER BY col LIMIT n`
- Aggregates: `COUNT, SUM, AVG, MIN, MAX`; `WHERE` filters rows, `HAVING` filters groups
- Joins: `INNER` (matches only), `LEFT/RIGHT` (keep one side), `FULL OUTER` (keep both) — combining tables on keys
- Subqueries and "for-all" (relational division) patterns: e.g., students who took *every* course — implemented with `NOT EXISTS ... NOT EXISTS` double negation
- Python + MySQL: `mysql.connector` → cursor → `execute(query)` → fetch into pandas (`pd.read_sql`)
- Beyond relational: web databases and NoSQL (document, key-value) trade schema rigidity for scale and flexibility — context for Unit VI

**The math.** Set theory underlies it: joins are set intersections/products, `UNION`/`INTERSECT`/`EXCEPT` are literal set operations, and "for-all" queries are universal quantification (∀) expressed via double negation (¬∃x ¬P(x)).

**Free resource:** [SQLBolt (interactive lessons)](https://sqlbolt.com/) — free in-browser SQL exercises from SELECT through joins

---

## Module 9: Databases & Data Storage

*(Extension module — builds on Module 8 and the ITE 451/651 web-database material.)*

**What it is.** How data is persisted and organized beyond a single table: transactional guarantees, storage layouts (row vs. column), NoSQL families, file formats, and the analytics storage stack (warehouse, lake, lakehouse, vector store).

**Why & when.** The storage choice shapes everything downstream. Operational systems (OLTP) need many small, safe reads/writes; analytics (OLAP) needs a few enormous scans. Schema flexibility trades against integrity; local files trade against distributed scale. Knowing the families tells you where your data should live — and why your queries are slow.

**How it's applied.**
- OLTP vs. OLAP: normalized row stores (MySQL, PostgreSQL — Module 8) for operations; columnar warehouses (Snowflake, BigQuery, Redshift) for analytics
- Dimensional modeling for analytics: star schema — a central *fact* table (events, transactions) joined to *dimension* tables (customer, product, date); deliberately denormalized for scan speed
- Normalization (1NF → 2NF → 3NF): store each fact once to eliminate update/insert/delete anomalies — the design discipline behind Module 8's keys
- Indexing: a B-tree index turns full scans into lookups; the cost is slower writes and extra storage — index the columns you filter and join on
- NoSQL families: document (MongoDB — JSON-like, flexible schema), key-value (Redis — caching, sessions), wide-column (Cassandra — massive write throughput), graph (Neo4j — relationships as first-class citizens)
- File formats for data science: CSV (universal, untyped, slow), JSON (nested, verbose), Parquet (columnar, compressed, typed — the analytics default; `pd.read_parquet`), Arrow/Feather (fast interchange)
- The analytics stack: data lake (raw files, schema-on-read) → warehouse (curated tables, schema-on-write) → lakehouse (both, e.g. Databricks/Delta); ETL (transform before load) vs. ELT (transform inside the warehouse)
- Vector databases (FAISS, Chroma, Pinecone): store embeddings, retrieve by similarity — the storage layer behind RAG (Module 32)

**The math.**
- ACID (Atomicity, Consistency, Isolation, Durability) vs. BASE (Basically Available, Soft state, Eventual consistency) — the transactional/NoSQL tradeoff
- CAP theorem: under a network Partition, a distributed store must choose Consistency or Availability — you can't have all three
- B-tree lookup: O(log n) vs. full scan O(n) — the same complexity logic as binary search (Module 7)
- Columnar advantage: scanning k of p columns reads ≈ k/p of the data, and homogeneous columns compress far better — why warehouses are columnar
- Normal forms: 1NF atomic values; 2NF no partial dependency on a composite key; 3NF no transitive dependencies (non-key → non-key)
- Vector search: exact nearest-neighbor is O(n·d); approximate indexes (e.g., HNSW graphs) reach ~O(log n) by trading a little recall for speed

**Free resource:** [What Is NoSQL? — AWS (article)](https://aws.amazon.com/nosql/) — concise tour of NoSQL families vs. relational, with use cases

---

## Module 10: Data Wrangling & Data Quality

**What it is.** Transforming raw, messy data into a clean, model-ready table: handling missing values, duplicates, outliers, and encoding.

**Why & when.** "Garbage in, garbage out." Models require numeric, complete, consistently-scaled input. This is routinely 60–80% of project time. Your MIS 769 HW1 formalized it as Data Quality Assessment; your Midterm Q11 began the same way ("the first thing I did was evaluate the dataset... clean it by removing irrelevant columns").

**How it's applied.**
- Missing values (`NaN`): detect with `df.isnull().sum()`; drop (`dropna`) if few/random, or impute — mean/median for numeric (median if skewed), mode for categorical. In your association-rules data, NaN simply meant "no item in that slot"— context determines meaning
- Duplicates: `df.duplicated().sum()` → `drop_duplicates()`
- Outliers: detect via IQR rule or z-score (|z| > 3); decide — error (fix/drop) vs. genuine extreme (often keep)
- Categorical encoding: `pd.get_dummies(df, drop_first=True)` — one-hot encoding. Dropping the first level avoids the dummy variable trap (perfect multicollinearity) and makes the dropped level the *reference category*: every dummy coefficient is interpreted relative to it (why you couldn't directly compare Northwest when Southeast was the reference — Assignment1_Regression Q9)
- Scaling: standardization z = (x − μ)/σ (`StandardScaler`) or min-max to [0,1]. Required by distance-based methods (KNN, K-Means, SVM); irrelevant to trees
- Feature engineering: creating informative columns (ratios, date parts, binned ages, text lengths)

**The math.** z-score: z = (x − μ)/σ — "how many standard deviations from the mean." Min-max: x′ = (x − min)/(max − min). One-hot: a categorical variable with k levels becomes k−1 binary columns.

**Free resource:** [Data Cleaning — Kaggle Learn (interactive course)](https://www.kaggle.com/learn/data-cleaning) — hands-on missing values, scaling, parsing, and inconsistent data

---

## Module 11: Data Visualization

**What it is.** Encoding data in visual form — position, length, color — so patterns become visible.

**Why & when.** Twin purposes: *exploration* (you finding patterns: skew, outliers, clusters) and *communication* (stakeholders understanding results). Anscombe's lesson: identical summary statistics can hide wildly different data — always plot.

**How it's applied (matplotlib / seaborn, used throughout your assignments):**
- Distribution of one variable: histogram (`plt.hist`, choose bins thoughtfully), boxplot, KDE
- Comparison across groups: side-by-side boxplots (survivor vs. non-survivor ages), bar charts
- Relationship between two numerics: scatter plot (Age vs. Fare), with regression line overlay
- Many pairwise relationships: correlation heatmap (`sns.heatmap(df.corr(), annot=True)`) — how you spotted multicollinearity in the Boston data (RAD×TAX = 0.91)
- Model diagnostics: residual plots, elbow plots, ROC curves (Modules 14, 23, 16)
- Principles: label axes, title everything, don't truncate bar-chart y-axes, use color meaningfully, prefer direct labeling over legends

**The math.** Histogram bin counts trade bias vs. variance (too few bins oversmooths, too many shows noise); KDE smooths with kernel bandwidth. Otherwise this module is design judgment, not formulas.

**Free resource:** [From Data to Viz (interactive guide)](https://www.data-to-viz.com/) — decision tree from your data type to the right chart, with code and caveats

---

# UNIT IV — Machine Learning: Core Concepts & Supervised Learning

## Module 12: The Machine Learning Workflow

**What it is.** The discipline around any model: framing the task, splitting data, fitting, evaluating, and diagnosing overfitting.

**Why & when.** Before any specific algorithm. The workflow — not the algorithm — is what makes results trustworthy. It appears identically in every modeling assignment you did.

**How it's applied.**
- Task types: **supervised** (labeled target y: regression if y numeric, classification if categorical) vs. **unsupervised** (no y: clustering, dimensionality reduction, association rules)
- Split: `train_test_split(X, y, test_size=0.3, random_state=42)` — fit on train, judge on test. You verified split fairness by comparing train/test feature means (MIS 761 A8 Q4)
- Overfitting: model memorizes training noise → great train score, poor test score. Your A8 Q7 decision tree: near-perfect on train, notably worse on test = classic overfitting. Underfitting: too simple for either set
- Bias–variance tradeoff: simple models = high bias (systematically wrong); complex models = high variance (unstable, noise-sensitive). Total error ≈ bias² + variance + irreducible noise. Control complexity via depth limits, regularization, or ensembles (Module 20)
- Cross-validation: k-fold CV (`cross_val_score`, cv=5) — rotate which fold is held out, average the scores; more reliable than a single split
- Hyperparameter tuning: `GridSearchCV` searches settings (k, max_depth, C) using CV — never tune on the test set
- Golden rule: the test set is touched once, at the end

**The math.** Expected prediction error decomposition: E[(y − ŷ)²] = Bias(ŷ)² + Var(ŷ) + σ². k-fold CV estimate: (1/k)Σ scoreᵢ.

**Free resource:** [Machine Learning Crash Course — Google (free course)](https://developers.google.com/machine-learning/crash-course) — the full workflow — splits, loss, generalization, overfitting — with visuals and quizzes

---

## Module 13: Model Validation Deep Dive

*(Extension module — goes beyond your coursework, filling gaps around Module 12.)*

**What it is.** Techniques that keep performance estimates honest beyond a basic split: three-way splits, stratification, learning curves, leakage prevention, and time-aware validation.

**Why & when.** Whenever a metric will drive a decision. Naive validation breaks quietly: imbalanced classes make random folds unrepresentative, heavy tuning overfits the test set, preprocessing fit on all data leaks information, and shuffling destroys temporal order. These are the failure modes behind models that score well offline and flop in production.

**How it's applied.**
- Train/validation/test three-way split: tune hyperparameters on *validation*, report once on *test*. In practice: `train_test_split` twice, or use CV on the training set as the validation role (what `GridSearchCV` does internally)
- Stratified k-fold: `StratifiedKFold` / `train_test_split(stratify=y)` preserves class proportions in every fold — the sensible default for classification (Titanic's ~62/38 died/survived split stays intact per fold)
- Learning curves: `sklearn.model_selection.learning_curve` — plot train & validation score vs. training-set size to diagnose whether more data would help
- Validation curves: score vs. one hyperparameter (k, max_depth, C) — visualizes the bias–variance sweep from Module 12
- Data leakage: fit scalers/imputers/encoders on the training fold only — wrap preprocessing in a `Pipeline` so CV refits it per fold. Also watch *target leakage*: features that encode the outcome (e.g., a column recorded after the label)
- Time-series CV: `TimeSeriesSplit` — expanding-window folds; never shuffle time; never train on the future to predict the past

**The math.**
- Stratification lowers the variance of the CV estimate by forcing each fold's class proportion ≈ the overall p (variance reduction by conditioning)
- Report CV as mean ± std across folds — the spread is your uncertainty about the score itself
- Learning-curve reading: both curves plateau low and close → high bias (more data won't help; add features/complexity); large persistent gap → high variance (more data or regularization helps)
- Test-set reuse: every peek at the test set is an implicit comparison; enough peeks and you've tuned to its noise — the multiple-comparisons problem in disguise
- Nested CV: outer loop estimates generalization, inner loop tunes — the unbiased way to tune *and* evaluate on limited data
- Leakage in math terms: validation assumes train and test rows are independent samples; any statistic computed using test rows (a scaling mean, an imputation) breaks that independence and biases scores upward

**Free resource:** [Cross-Validation User Guide — scikit-learn (docs)](https://scikit-learn.org/stable/modules/cross_validation.html) — stratified k-fold, TimeSeriesSplit, and leakage warnings, straight from the source

---

## Module 14: Linear Regression

**What it is.** Modeling a numeric target as a weighted sum of features: the line (or hyperplane) that best fits the data.

**Why & when.** The default first model for predicting quantities (prices, charges, demand) and for *explaining* relationships — its coefficients are directly interpretable. Use when relationships are roughly linear; check assumptions before trusting inference.

**How it's applied.**
- `statsmodels.OLS` (rich statistical summary — your MIS 761 A6, Boston housing) and `sklearn.LinearRegression` (prediction pipeline — Assignment1_Regression, insurance charges)
- Coefficient reading: "+1 child → +$440 in charges"; dummy coefficients read against the reference category ("smoker=yes adds ...", "Northwest vs. Southeast reference")
- Assumption checks (A6): linearity (scatter/residual plots), normal residuals (histogram/Q-Q — you found it *not* met), homoscedasticity (residual-vs-fitted plot — not met), multicollinearity (correlation heatmap + VIF — TAX and RAD near 10, concerning)
- Prediction: `model.predict(new_X)` → your $36,208 charge estimate

**The math.**
- Model: y = β₀ + β₁x₁ + … + βₚxₚ + ε
- OLS objective: choose β to minimize SSE = Σ(yᵢ − ŷᵢ)². Closed form: β̂ = (XᵀX)⁻¹Xᵀy (linear algebra from Module 2!)
- Simple-regression slope: β₁ = cov(x,y)/var(x) = r·(sᵧ/sₓ)
- R²: proportion of variance explained = 1 − SSE/SST, SST = Σ(yᵢ − ȳ)². Adjusted R² penalizes added predictors: 1 − (1−R²)(n−1)/(n−p−1)
- Error metrics: MSE = SSE/n; RMSE = √MSE (in y's units); MAE = Σ|yᵢ−ŷᵢ|/n (robust to outliers)
- Assumptions (for valid inference): (1) linearity, (2) independent errors, (3) homoscedasticity — constant error variance, (4) normal errors, (5) no perfect multicollinearity
- VIF for predictor j: VIF = 1/(1 − Rⱼ²), where Rⱼ² comes from regressing xⱼ on the other predictors. VIF > 5–10 → serious multicollinearity: coefficients become unstable with inflated standard errors (predictions can still be fine — it's the *interpretation* that breaks)
- Coefficient t-tests: t = β̂/SE(β̂); p < α → predictor is statistically significant

**Free resource:** [Linear Regression, Clearly Explained — StatQuest (video)](https://www.youtube.com/watch?v=nk2CQITm_eo) — least squares, R², and p-values for regression, step by step

---

## Module 15: Logistic Regression

**What it is.** Regression adapted to binary outcomes: it models the *probability* of class 1 by passing a linear combination through the sigmoid function.

**Why & when.** The workhorse baseline for binary classification (survived/died, churn/stay, fraud/legit) — fast, probabilistic, and interpretable through odds ratios. Your MIS 761 A7 (Titanic survival) is a complete case study.

**How it's applied.**
- `statsmodels.Logit` for inference (coefficients, p-values, pseudo-R², LLR test — your A7) and `sklearn.LogisticRegression` for prediction pipelines (Assignment2, Midterm)
- Reading your A7 results: male coefficient −2.64 (p < 0.001) → odds ratio e^(−2.64) ≈ 0.07 — men had ~93% lower odds of survival; Fare coefficient 0.0014 → OR ≈ 1.0014, statistically detectable but practically tiny per $1
- Model-level checks: pseudo-R² = 0.34 (fit quality), LLR p-value 3.4e-66 (model significant overall), converged in 6 iterations (it's fit by iterative optimization, not a closed form)
- Significant predictors at α = 0.05: Pclass, Age, Sex — the variables that mattered

**The math.**
- Sigmoid: p = 1/(1 + e^(−z)), z = β₀ + β₁x₁ + … — squashes any real z into (0,1)
- Equivalent form: log-odds are linear: ln(p/(1−p)) = β₀ + β₁x₁ + …
- Odds = p/(1−p). Odds ratio for xⱼ: OR = e^(βⱼ) — multiplicative change in odds per one-unit increase. OR < 1 decreases odds, > 1 increases
- Fitting: maximum likelihood estimation (MLE) — maximize L = Π pᵢ^(yᵢ)(1−pᵢ)^(1−yᵢ), equivalently minimize log-loss: −Σ[yᵢ ln pᵢ + (1−yᵢ)ln(1−pᵢ)]. Solved iteratively by gradient-based methods (Module 1) — hence "6 iterations"
- McFadden pseudo-R²: 1 − (ln L_model / ln L_null) — not variance-explained like OLS R²; 0.2–0.4 already indicates a good fit
- LLR test: compares full model vs. intercept-only; small p → model adds real information
- Decision rule: predict class 1 if p ≥ threshold (default 0.5; move it to trade precision vs. recall — Module 16)

**Free resource:** [Logistic Regression — StatQuest (video)](https://www.youtube.com/watch?v=yIYKR4sgzI8) — how the sigmoid and log-odds connect back to linear regression

---

## Module 16: Evaluating Classifiers

**What it is.** The metric toolkit for judging classification models — because accuracy alone misleads.

**Why & when.** Every classification assignment ends here. Essential whenever classes are imbalanced (72% of Titanic passengers died — always predicting "died" gets 72% accuracy while learning nothing) or when error types have different costs (missing a fraud ≠ flagging a legit transaction).

**How it's applied.**
- `confusion_matrix`, `classification_report` (precision/recall/F1 per class), `accuracy_score` — used in A8, A9, Assignment1_Classification, Assignment2, Midterm
- Your A8 pattern-reading: non-survivors had higher F1 than survivors → the model predicts the majority class better; comparable train vs. test scores → no overfitting evidence
- Comparing models fairly: your Assignment2 verdict — logistic regression scored higher accuracy than Naive Bayes, but that alone doesn't make it "better"; consider stability, calibration, and error structure

**The math.**
- Confusion matrix: TP, TN, FP (Type I), FN (Type II)
- Accuracy = (TP+TN)/all — fine only when classes are balanced and errors cost the same
- Precision = TP/(TP+FP) — "of predicted positives, how many were right?" Optimize when false positives are costly (spam filters)
- Recall (sensitivity) = TP/(TP+FN) — "of actual positives, how many did we catch?" Optimize when misses are costly (disease screening)
- F1 = 2·(precision·recall)/(precision+recall) — harmonic mean; punishes imbalance between the two. Your go-to single number per class
- Specificity = TN/(TN+FP)
- ROC curve: TPR vs. FPR across all thresholds; AUC = P(random positive ranked above random negative); 0.5 = coin flip, 1.0 = perfect
- Precision–recall tradeoff: raising the decision threshold raises precision, lowers recall — pick per the business cost of each error

**Free resource:** [Classification Metrics — Google ML Crash Course (course unit)](https://developers.google.com/machine-learning/crash-course/classification) — interactive thresholds, precision/recall, and ROC/AUC

---

## Module 17: Naive Bayes

**What it is.** A probabilistic classifier that applies Bayes' theorem with a deliberately "naive" assumption: features are independent given the class.

**Why & when.** Extremely fast, works with little data, and shines on high-dimensional problems — especially text (its classic home: spam filtering, your Midterm text classification). A strong baseline before heavier models.

**How it's applied.**
- `GaussianNB` for continuous features (your Assignment2, Midterm — 72.8% accuracy → 27.2% error rate); `MultinomialNB` for word counts; `BernoulliNB` for binary features
- Head-to-head with logistic regression (Assignment2): NB trained instantly and generalized steadily; logistic scored higher accuracy — your conclusion: neither dominates universally, match the tool to the data

**The math.**
- Goal: pick class c maximizing the posterior P(c|x₁,…,xₚ)
- Bayes: P(c|x) ∝ P(c)·P(x|c). The naive step factorizes the likelihood: P(x₁,…,xₚ|c) = Π P(xᵢ|c) — independence given class
- Decision rule: ĉ = argmax_c P(c)·Π P(xᵢ|c) (computed in log space to avoid underflow: argmax log P(c) + Σ log P(xᵢ|c))
- Gaussian NB: P(xᵢ|c) modeled as Normal(μᵢ,c, σᵢ,c) estimated per feature per class
- Multinomial NB with Laplace smoothing: P(word|c) = (count + 1)/(total + |V|) — the +1 prevents a single unseen word from zeroing the whole product
- Why it works despite the false assumption: classification needs only the *argmax* to be right, not the probabilities to be calibrated

**Free resource:** [Naive Bayes, Clearly Explained — StatQuest (video)](https://www.youtube.com/watch?v=O2L2Uv9pdDA) — multinomial NB built by hand on a spam example, pseudocounts included

---

## Module 18: K-Nearest Neighbors (KNN)

**What it is.** A "lazy" learner: no training phase — to classify a new point, find the k closest training points and take a majority vote (or average, for regression).

**Why & when.** When the decision boundary is irregular and you have a modest, well-scaled dataset. Intuitive baseline; also the conceptual bridge to similarity search in embeddings (Module 30). Weakens with many features or big data (every prediction scans the training set).

**How it's applied.**
- `KNeighborsClassifier(n_neighbors=k)` — your MIS 761 A9, alongside decision trees on Titanic
- **Scale first** (`StandardScaler`) — otherwise the largest-unit feature (Fare) silently dominates the distance
- Choose k by validation curve: small k → jagged boundary, overfits; large k → oversmoothed, underfits. Odd k avoids ties in binary problems

**The math.**
- Euclidean distance: d(a,b) = √Σ(aᵢ−bᵢ)²; Manhattan: Σ|aᵢ−bᵢ|; Minkowski generalizes both
- Prediction: ŷ = mode{yᵢ : i ∈ N_k(x)} (classification) or mean (regression); optionally weight votes by 1/d
- k controls bias–variance directly: k=1 → zero training error, high variance; k=n → predicts the global majority, high bias
- Curse of dimensionality: in high dimensions all pairwise distances converge → "nearest" loses meaning; reduce dimensions first (Module 24)

**Free resource:** [K-Nearest Neighbors, Clearly Explained — StatQuest (video)](https://www.youtube.com/watch?v=HVXime0nQeI) — short and complete: the algorithm and how to pick k

---

## Module 19: Decision Trees

**What it is.** A model that predicts by asking a sequence of yes/no questions about features, forming a flowchart from root to leaf.

**Why & when.** When interpretability matters — the tree *is* the explanation. Handles nonlinearity and feature interactions natively, needs no scaling, mixes numeric and categorical inputs. Caveat: a single deep tree overfits readily (your A8 Q7 evidence), which motivates ensembles (Module 20).

**How it's applied.**
- `DecisionTreeClassifier(max_depth=…)` — A8, A9, Assignment1_Classification (where Age emerged as the most influential feature)
- Visualize with `plot_tree`; read feature importances with `.feature_importances_`
- Control overfitting: `max_depth`, `min_samples_split`, `min_samples_leaf`, or post-pruning (`ccp_alpha`)
- Your empirical arc: unconstrained tree → near-perfect train / degraded test (overfit); depth-limited tree → balanced performance

**The math.**
- Gini impurity of a node: G = 1 − Σ pₖ² (0 = pure, max at uniform mix). Entropy alternative: H = −Σ pₖ log₂ pₖ
- Split criterion: choose the feature & threshold maximizing impurity reduction: ΔG = G_parent − (n_L/n)G_L − (n_R/n)G_R (with entropy this is "information gain")
- Greedy recursive partitioning: best split at each node, no lookahead; stop at purity, depth limit, or minimum samples
- Feature importance: total impurity reduction contributed by each feature, summed over its splits, normalized
- Regression trees: split to minimize within-node variance; leaf predicts the node mean

**Free resource:** [Decision and Classification Trees, Clearly Explained — StatQuest (video)](https://www.youtube.com/watch?v=_L39rN6gz7Y) — builds a tree from scratch with Gini impurity

---

## Module 20: Ensemble Methods — Bagging, Random Forests, Boosting

**What it is.** Combining many weak or unstable models into one stronger, steadier predictor.

**Why & when.** Single trees are high-variance — small data changes reshape them. Averaging hundreds of diverse trees cancels the noise. Ensembles are the go-to for strong tabular-data performance. Your A9 bagging experiment is the proof: 250 bagged trees beat one tree, and constraining base learners to max_depth=1 (stumps) markedly stabilized results — your own words: a single deep tree is "prone to overfit," and the stump constraint made "a great difference in the stability of the outcome."

**How it's applied.**
- `BaggingClassifier(estimator=DecisionTreeClassifier(…), n_estimators=250)` (A9)
- `RandomForestClassifier` — bagging + random feature subsets per split; also yields robust feature importances
- Boosting: `GradientBoostingClassifier`, XGBoost/LightGBM — sequential error-correction; typically the strongest tabular models, but more tunable and more overfit-prone than forests

**The math.**
- Bootstrap: sample n rows *with replacement* → each replica omits ~36.8% of rows (since (1−1/n)ⁿ → e⁻¹); the omitted "out-of-bag" rows give a free validation estimate
- Bagging = **b**ootstrap **agg**regating: train a model per replica; aggregate by vote (classification) or mean (regression)
- Variance reduction: averaging B estimators with pairwise correlation ρ gives Var = ρσ² + (1−ρ)σ²/B — the reason random forests decorrelate trees by considering only a random subset of features (≈√p) at each split
- Boosting (conceptually): F_m(x) = F_{m−1}(x) + η·h_m(x), where each new weak learner h_m fits the residuals/gradients of the current ensemble; learning rate η shrinks each step
- Bagging attacks **variance**; boosting attacks **bias** — opposite medicine, both ensembles

**Free resource:** [Random Forests Part 1 — StatQuest (video)](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) — bootstrapping, bagging, and out-of-bag error, visually

---

## Module 21: Support Vector Machines (SVM)

**What it is.** A classifier that finds the separating boundary with the *maximum margin* — the widest possible gap between classes — and can bend that boundary via kernels.

**Why & when.** Strong on medium-sized, high-dimensional data (your ITE 622 pairing: SVM + OCR — pixel features, classic territory). Choose it when a clear margin exists or when kernels can untangle nonlinear structure. Requires feature scaling; slows on very large datasets.

**How it's applied.**
- `SVC(kernel='linear'|'rbf', C=…, gamma=…)`; scale features first
- Tune C and gamma with `GridSearchCV`
- Only the *support vectors* — points on or inside the margin — define the boundary; the rest of the data could vanish without changing the model

**The math.**
- Hyperplane: wᵀx + b = 0; margin width = 2/‖w‖
- Hard-margin objective: minimize ½‖w‖² subject to yᵢ(wᵀxᵢ + b) ≥ 1
- Soft margin: minimize ½‖w‖² + C·Σξᵢ — slack ξᵢ permits violations; C prices them (large C → strict fit, risk of overfit; small C → wider margin, more tolerance)
- Kernel trick: replace dot products with k(xᵢ,xⱼ) to operate in an implicit higher-dimensional space where classes become separable — without computing coordinates there. RBF kernel: k(x,x′) = e^(−γ‖x−x′‖²); γ sets locality (large γ → wiggly boundary)

**Free resource:** [Support Vector Machines Part 1 — StatQuest (video)](https://www.youtube.com/watch?v=efR1C6CvhmE) — margins, soft margins, and the kernel trick intuition

---

## Module 22: Neural Networks & Deep Learning

**What it is.** Layers of simple units ("neurons"), each computing a weighted sum passed through a nonlinear activation; stacking layers lets the network learn arbitrarily complex functions. "Deep" = many layers.

**Why & when.** When patterns are too complex for linear models or trees — images, audio, text — and data is plentiful. Deep learning powers OCR, embeddings (Module 30), and LLMs (Module 32). Cost: data-hungry, compute-heavy, less interpretable. Your ITE 622 arc: neural networks → deep learning → TensorFlow → Keras.

**How it's applied.**
- Keras: `Sequential([Dense(64, activation='relu'), Dense(1, activation='sigmoid')])`; `compile(optimizer='adam', loss='binary_crossentropy')`; `fit(..., epochs, batch_size, validation_split)`
- Architecture by task: output layer sigmoid + binary cross-entropy (binary), softmax + categorical cross-entropy (multiclass), linear + MSE (regression)
- Fight overfitting: dropout, early stopping (watch validation loss), L2 weight decay, more data
- Specialized layers: convolutional (images — local filters, shared weights), recurrent (sequences)

**The math.**
- One neuron: a = φ(wᵀx + b). Activations: ReLU φ(z) = max(0,z) (default hidden), sigmoid (binary output), softmax(zᵢ) = e^(zᵢ)/Σe^(zⱼ) (multiclass output). Nonlinearity is essential — without it, stacked layers collapse into one linear map (Module 2)
- Forward pass: h⁽ˡ⁾ = φ(W⁽ˡ⁾h⁽ˡ⁻¹⁾ + b⁽ˡ⁾) — matrix multiplication all the way down
- Loss: binary cross-entropy = the same log-loss as logistic regression (Module 15); logistic regression *is* a one-neuron network
- Backpropagation: the chain rule (Module 1) applied layer-by-layer to get ∂L/∂w for every weight
- Gradient descent: w ← w − α·∂L/∂w; stochastic/mini-batch variants estimate the gradient on small batches; Adam adapts per-weight step sizes
- Universal approximation theorem: one sufficiently wide hidden layer can approximate any continuous function — depth just makes it efficient

**Free resource:** [Neural Networks — 3Blue1Brown (video series)](https://www.3blue1brown.com/topics/neural-networks) — gradient descent and backpropagation, beautifully animated

---

# UNIT V — Unsupervised Learning

## Module 23: Clustering (K-Means)

**What it is.** Grouping unlabeled observations so members of a group are similar and groups are distinct. K-Means partitions data into k clusters around centroids.

**Why & when.** No target variable — you're discovering structure: customer segments, neighborhood tiers, document groups. Your MIS 761 A10 (housing) named the discoveries — cluster 0 "lower-value neighborhoods," cluster 1 "higher-value" — then showed k=3 usefully split out a middle tier. Also used at scale in your Spark HW2.

**How it's applied.**
- Scale features → `KMeans(n_clusters=k, random_state=…)` → `.labels_`, `.cluster_centers_`
- Choose k: elbow plot (inertia vs. k — pick the bend) and silhouette score
- *Interpret* clusters by comparing feature means per cluster, then name them (your A10 workflow)
- Cluster-then-model (A10 Q8–Q12): fit a separate regression per cluster and compare with the global baseline — you found cluster 0's model underperformed baseline (lower R²) while cluster 1's ran higher on both R² and MSE — segmenting changes what's learnable in each stratum

**The math.**
- Objective: minimize inertia (WCSS) = Σₖ Σ_{x∈Cₖ} ‖x − μₖ‖²
- Lloyd's algorithm: (1) place k centroids, (2) assign each point to nearest centroid, (3) recompute centroids as cluster means, (4) repeat until stable. Each step lowers inertia → converges, but possibly to a local optimum — hence `n_init` restarts
- Elbow method: inertia always falls as k rises; the "elbow" marks diminishing returns
- Silhouette: s = (b − a)/max(a,b), a = mean intra-cluster distance, b = mean nearest-other-cluster distance; s ∈ [−1,1], higher = crisper clustering
- Limits: assumes roughly spherical, similar-size clusters; sensitive to scaling and outliers. Alternatives: hierarchical clustering (dendrograms), DBSCAN (density-based, finds arbitrary shapes, flags noise)

**Free resource:** [K-Means Clustering — StatQuest (video)](https://www.youtube.com/watch?v=4b5d3muPQmA) — the algorithm plus elbow plots in 8 minutes

---

## Module 24: Dimensionality Reduction (PCA)

**What it is.** Compressing many correlated features into a few new axes ("principal components") that retain most of the variance.

**Why & when.** High-dimensional data is hard to visualize, slow to model, and prone to the curse of dimensionality (Module 18) and multicollinearity (Module 14). PCA fixes all three: visualization (project to 2D), noise reduction, and decorrelated inputs. Part of your ITE 622 unsupervised unit.

**How it's applied.**
- Standardize first (PCA chases variance — unscaled units hijack it) → `PCA(n_components=…)` → `fit_transform`
- Pick components by `explained_variance_ratio_`: keep enough for ~80–95% cumulative variance (scree plot)
- Interpret loadings: each component is a weighted blend of original features
- Pipeline uses: PCA → K-Means (cleaner clusters), PCA → regression (kills multicollinearity), PCA → 2D scatter (see structure)

**The math.**
- Center the data; compute covariance matrix C = XᵀX/(n−1)
- Eigendecomposition (Module 2): C's eigenvectors are the principal components (orthogonal directions of maximal variance); eigenvalues λᵢ are the variance along each
- Explained variance ratio of PC i: λᵢ/Σλⱼ
- Projection: Z = XW, where W's columns are the top-d eigenvectors — a linear map to d dimensions with minimal information (variance) loss
- Equivalently computed via SVD: X = UΣVᵀ, components = columns of V

**Free resource:** [PCA Step-by-Step — StatQuest (video)](https://www.youtube.com/watch?v=FgakZw6K1QQ) — principal components, loadings, and scree plots worked in full

---

## Module 25: Association Rule Mining

**What it is.** Discovering "if–then" co-occurrence rules in transactions: {A} → {B}, "baskets containing A tend to contain B."

**Why & when.** Market-basket analysis: product placement, cross-sell recommendations, bundling. Unsupervised — no target, just patterns in what appears together. Your Assignment3 groceries analysis is the template.

**How it's applied.**
- Reshape transactions (your 10 × 675 wide layout, NaN = empty slot) → one-hot basket matrix (`TransactionEncoder`)
- `mlxtend`: `apriori(df, min_support=…)` → frequent itemsets → `association_rules(metric='confidence'|'lift', min_threshold=…)`
- Reading your A3 rule: support 0.0088 → the pair appears in ~0.88% of all baskets; confidence → how reliably the antecedent leads to the consequent; lift > 1 → genuinely associated, not coincidence

**The math.**
- Support(A→B) = P(A ∩ B) = count(A and B together)/total transactions — how common
- Confidence(A→B) = P(B|A) = support(A∪B)/support(A) — how reliable
- Lift(A→B) = confidence/P(B) = P(A∩B)/(P(A)P(B)) — how surprising: lift = 1 → independent (Module 3!); > 1 positive association; < 1 substitution. Confidence alone misleads when B is just popular — lift corrects for that
- Apriori principle: every subset of a frequent itemset is frequent — so prune candidates containing any infrequent subset. This turns an exponential search (2ⁿ itemsets) into a tractable one

**Free resource:** [Apriori — mlxtend User Guide (docs + tutorial)](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/) — the exact library and workflow from your Assignment3, with runnable examples

---

# UNIT VI — Big Data & Distributed Computing

## Module 26: Big Data Concepts

**What it is.** Data too large, fast, or varied for a single machine — and the architectural shift from scaling *up* (bigger computer) to scaling *out* (many computers).

**Why & when.** When datasets outgrow RAM, when processing must parallelize, when data arrives as streams. Framing from your MIS 769 weeks 1–2: "when data gets too big for one computer, we split the work across many."

**How it's applied.**
- The V's: **Volume** (size), **Velocity** (speed of arrival), **Variety** (structured tables + text + images + logs) — plus Veracity (quality) and Value
- Distributed storage: HDFS — files split into blocks, replicated (typically 3×) across nodes, so hardware failure loses nothing
- Ecosystem literacy: Hadoop (storage + batch), Spark (fast general compute), NoSQL stores (Module 8's tail), cloud object storage; data lake (raw, schema-on-read) vs. warehouse (curated, schema-on-write)
- Data source plumbing from your HW1: Colab ↔ Kaggle/HuggingFace APIs, loading large datasets responsibly (sampling, chunking, dtype management)

**The math.** Mostly systems reasoning, but: replication factor r means storage cost r× for fault tolerance; Amdahl's law caps parallel speedup — speedup ≤ 1/(s + (1−s)/N) where s is the serial fraction — why coordination overhead means N machines never give N× throughput.

**Free resource:** [What Is Big Data? — AWS (article)](https://aws.amazon.com/what-is/big-data/) — the V's, storage/processing architecture, and ecosystem in plain language

---

## Module 27: MapReduce & Apache Spark

**What it is.** MapReduce: a programming model that expresses computation as two parallelizable phases — *map* (transform each record independently) and *reduce* (aggregate by key). Spark: the modern in-memory engine built on the same idea, with far less disk I/O and richer APIs.

**Why & when.** When one machine can't process the data in reasonable time. Word counts over millions of documents, aggregations over billions of rows, distributed ML (your Spark HW2 ran K-Means — Module 23 — on a cluster).

**How it's applied.**
- MapReduce in Python first (MIS 769 week 3) to internalize the pattern: map emits (key, value) pairs → shuffle groups by key → reduce aggregates each group. Canonical word count: map word → (word, 1); reduce sums
- PySpark on Colab (HW2): `SparkSession`, DataFrames, transformations (`filter`, `groupBy`, `select`) vs. actions (`count`, `collect`, `show`)
- Partitions: data splits across executors; you measured how partition count changes wall-clock performance — too few = idle cores, too many = scheduling overhead
- Lazy evaluation: transformations only build a DAG (execution plan); nothing runs until an action forces it — letting Spark optimize the whole pipeline
- MLlib for distributed model training

**The math.** The formal skeleton: map: (k₁,v₁) → list(k₂,v₂); reduce: (k₂, list(v₂)) → list(v₃). Aggregations must be associative & commutative to parallelize safely (sums and counts qualify; medians don't decompose — they need workarounds). Shuffle is the expensive step: O(data) network movement — good distributed design minimizes shuffles.

**Free resource:** [Quick Start — Apache Spark (docs tutorial)](https://spark.apache.org/docs/latest/quick-start.html) — official hands-on intro to Spark DataFrames, transformations, and actions

---

# UNIT VII — Text Analytics, NLP & Large Language Models

## Module 28: Text Preprocessing

**What it is.** Converting raw text into normalized tokens: tokenization, lowercasing, stopword removal, stemming/lemmatization.

**Why & when.** Text is unstructured; models need consistent units. Preprocessing shrinks vocabulary and merges variants ("Running", "runs" → "run"). Your MIS 769 HW3 stressed the *why*, not just the how — including when cleaning **hurts**: sentiment analysis can break if you strip negations ("not") as stopwords; NER needs capitalization; domain terms need custom stopword lists (you built one for your data).

**How it's applied.**
- NLTK & spaCy: `word_tokenize`, stopword lists, `PorterStemmer` (fast, crude: "studies"→"studi") vs. `WordNetLemmatizer`/spaCy lemmas (dictionary-informed: "studies"→"study")
- Standard pipeline (your Midterm Q11 followed it): clean → lowercase → strip punctuation (`string.punctuation`)/digits → tokenize → remove stopwords → stem/lemmatize → vectorize (Module 29) → classify
- The full arc as taught: Raw Input → Preprocess → Tokenize → Embed → Classify
- Judgment call each time: match preprocessing aggressiveness to the downstream task

**The math.** Zipf's law: word frequency ∝ 1/rank — a handful of words dominate all text (why stopword removal cuts volume so much while losing little signal). Vocabulary reduction directly shrinks the feature space of Module 29.

**Free resource:** [Advanced NLP with spaCy (free interactive course)](https://course.spacy.io/en/) — tokenization, pipelines, and linguistic features in the browser

---

## Module 29: Text Vectorization & Text Classification

**What it is.** Turning documents into numeric vectors — Bag of Words counts or TF-IDF weights — so standard classifiers can run on text.

**Why & when.** Every model in Unit IV needs numbers. BoW/TF-IDF remains the fast, strong baseline for document classification: spam detection, ticket routing, sentiment. Your Midterm Q11 is the end-to-end case: preprocess → `CountVectorizer` → Naive Bayes & logistic regression → evaluate.

**How it's applied.**
- `CountVectorizer` (raw counts) / `TfidfVectorizer` (weighted); key knobs: `max_features`, `min_df`/`max_df` (drop too-rare/too-common terms), `ngram_range=(1,2)` to capture short phrases ("not good")
- Output is a sparse matrix: docs × vocabulary, mostly zeros — pair naturally with MultinomialNB (Module 17) or logistic regression (Module 15)
- Evaluate with the Module 16 toolkit (per-class F1 especially, since text classes are usually imbalanced)

**The math.**
- Term frequency: tf(t,d) = count of t in d (often normalized by document length)
- Inverse document frequency: idf(t) = ln(N/df(t)) — rare-across-corpus words score high; ubiquitous words → ~0
- TF-IDF: w(t,d) = tf(t,d) × idf(t) — "important here, distinctive overall"
- Vectors are usually L2-normalized; document similarity = cosine similarity: cos(a,b) = a·b/(‖a‖‖b‖) (the Module 2 dot product)
- Limits: order lost ("dog bites man" = "man bites dog"), no synonym awareness ("car" ⊥ "automobile") — the gap embeddings fill

**Free resource:** [Text Classification Guide — Google Developers (article series)](https://developers.google.com/machine-learning/guides/text-classification) — end-to-end: gather → explore → vectorize → train → tune, mirroring your Midterm pipeline

---

## Module 30: Word Embeddings (Word2Vec)

**What it is.** Dense, low-dimensional vectors (typically 100–300 numbers) for words, learned so that words in similar contexts land near each other. The opposite of one-hot sparsity.

**Why & when.** One-hot vectors are huge and orthogonal — "king" is exactly as unrelated to "queen" as to "toaster." Embeddings encode *meaning as geometry*: similarity, analogy, clustering of concepts. Your MIS 769 HW4: trained your own Word2Vec, built business analogies, and — importantly — documented where embeddings **fail** (rare words, small corpora, polysemy: one vector must average all senses of "bank").

**How it's applied.**
- `gensim.models.Word2Vec(sentences, vector_size, window, min_count, sg=0|1)`
- Query: `.most_similar('word')`, analogies: `most_similar(positive=['king','woman'], negative=['man'])` → "queen"
- Distributional hypothesis in action: "you shall know a word by the company it keeps"
- Downstream: average word vectors → document vector for classification; embeddings feed neural networks (Module 22) and are the conceptual ancestor of LLM token embeddings (Module 32)

**The math.**
- One-hot: |V|-dimensional, single 1 — all pairwise similarity 0. Dense: d ≪ |V|, real-valued, similarity meaningful
- Two training architectures: **CBOW** (predict center word from context — fast, good for frequent words) and **Skip-gram** (predict context from center word — better for rare words, small corpora)
- Skip-gram objective: maximize Σ log P(context|center), with P(o|c) = softmax(u_oᵀv_c) over the vocabulary; negative sampling approximates the expensive softmax by discriminating true context words from k random ones
- Similarity: cosine (Module 2). Analogy arithmetic: v(king) − v(man) + v(woman) ≈ v(queen) — relationships become vector offsets
- Failure modes to check: out-of-vocabulary words, unstable vectors for low-frequency terms (below `min_count`), corpus bias baked into geometry

**Free resource:** [The Illustrated Word2Vec — Jay Alammar (article)](https://jalammar.github.io/illustrated-word2vec/) — the definitive visual walkthrough of embeddings, CBOW/skip-gram, and negative sampling

---

## Module 31: Topic Modeling & Advanced NLP

**What it is.** Unsupervised discovery of themes in a document collection — classically LDA; in your coursework, BERTopic (embedding-based, MIS 769 HW5).

**Why & when.** You have thousands of unlabeled documents (reviews, tickets, articles) and need to know *what they're about* without reading them all. It's the clustering (Module 23) of the text world; feeds dashboards, routing, and research literature maps.

**How it's applied.**
- BERTopic pipeline: embed documents (transformer sentence embeddings) → reduce dimensions (UMAP) → cluster (HDBSCAN) → label each cluster with its class-distinctive terms (c-TF-IDF)
- Inspect: topic keyword lists, representative documents, topic frequency over time; merge/prune topics to taste
- Classical alternative — LDA: each document = mixture of topics, each topic = distribution over words; choose topic count k, inspect top words per topic

**The math.**
- LDA is a generative Bayesian model: for each document, draw topic proportions θ ~ Dirichlet(α); for each word, draw a topic z ~ Multinomial(θ), then a word w ~ Multinomial(β_z). Fitting inverts this: infer θ and β from observed words (via Gibbs sampling or variational inference) — Bayes' theorem (Module 3) at scale
- c-TF-IDF (BERTopic): TF-IDF computed treating each *cluster* as one concatenated document — surfacing words distinctive to each topic
- Coherence scores (e.g., c_v) rate topics by whether their top words co-occur — the quality check beyond eyeballing

**Free resource:** [BERTopic Documentation (docs + tutorials)](https://maartengr.github.io/BERTopic/) — the exact tool from your HW5 — algorithm explanation plus quick-start code

---

## Module 32: Large Language Models — Transformers, RAG & Prompt Engineering

**What it is.** Neural networks (Module 22) scaled to billions of parameters, built on the transformer architecture, trained to predict the next token over vast text corpora — and the applied toolkit around them: embeddings, retrieval-augmented generation, prompting, and fine-tuning. The back half of MIS 769 (weeks 7–11: LLM types & basics, embeddings & attention, RAG systems, prompt engineering & LoRA).

**Why & when.** LLMs handle open-ended language tasks — summarization, extraction, Q&A, generation — without task-specific training data. Use RAG when the model must answer from *your* documents; prompt engineering when behavior needs steering; fine-tuning when a model must deeply specialize.

**How it's applied.**
- Model landscape: encoder models (BERT — understanding/embedding), decoder models (GPT — generation), instruction-tuned chat models
- Sentence embeddings (your HW6): encode texts → vectors; semantic search by cosine similarity — Module 30 upgraded from words to passages
- RAG (your HW7): chunk documents → embed → store in a vector database → at query time, embed the question, retrieve top-k similar chunks, stuff them into the prompt → the LLM answers *grounded in your data*. Cures hallucination on private/current knowledge without retraining
- Prompt engineering (HW8): clear instructions, role framing, few-shot examples, requested output format, chain-of-thought ("think step by step"); iterate empirically
- LoRA fine-tuning: freeze the pretrained weights, train small low-rank adapter matrices — specialization at a sliver of full fine-tuning cost
- Practice: ethics & privacy awareness (your week 9 lecture) — data leakage, bias, verification of outputs

**The math.**
- Tokens → embedding vectors + positional encodings (order information)
- Self-attention, the transformer's core: Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V — every token computes relevance weights (query·key dot products — Module 2 again) over all other tokens and takes a weighted sum of their values. Multi-head attention runs several in parallel; stacked layers build contextual meaning ("bank" near "river" ≠ "bank" near "loan")
- Training objective: next-token prediction — minimize cross-entropy (Module 15's log-loss) over the corpus
- Generation: sample from the output distribution; temperature scales logits (low → deterministic, high → diverse)
- LoRA: replace a weight update ΔW (d×d) with BA where B is d×r, A is r×d, rank r ≪ d — trainable parameters drop from d² to 2dr
- RAG retrieval: argmax over cosine similarity between query and chunk embeddings

**Free resource:** [The Illustrated Transformer — Jay Alammar (article)](https://jalammar.github.io/illustrated-transformer/) — the classic visual explanation of attention, the architecture behind every LLM

---

# Appendix A — Suggested Study Sequence

| Phase | Modules | Theme |
|---|---|---|
| 1. Foundations | 1–3, 7 | Calculus, linear algebra, probability + Python fluency |
| 2. Statistics | 4–6 | Describe, infer, test |
| 3. Data craft | 8–11 | SQL, storage, wrangling, visualization |
| 4. Core ML | 12–16 | Workflow, validation, regression, logistic, evaluation |
| 5. Model zoo | 17–22 | NB, KNN, trees, ensembles, SVM, neural nets |
| 6. Unsupervised | 23–25 | Clustering, PCA, association rules |
| 7. Scale | 26–27 | Big data, Spark |
| 8. Language | 28–32 | Text → embeddings → topics → LLMs |

# Appendix B — Recurring Mathematical Threads

- **The dot product** (Module 2) reappears as: correlation, regression fitting, cosine similarity, SVM boundaries, attention scores
- **Bayes' theorem** (Module 3) reappears as: Naive Bayes, LDA topic models, the logic of p-values
- **Gradient descent** (Module 1) fits: logistic regression, SVMs, neural networks, Word2Vec, LLMs
- **The bias–variance tradeoff** (Module 12) explains: k in KNN, tree depth, bagging vs. boosting, regularization, dropout
- **Log-loss / cross-entropy** (Module 15) is the training objective of: logistic regression, neural classifiers, and LLM pretraining

# Appendix C — Concept-to-Coursework Index

| Concept | Where you practiced it |
|---|---|
| Derivatives, integrals, compounding | MIS 761 A1 (sympy) |
| Probability types | MIS 761 A2 (airline problems) |
| EDA, skew, correlation | MIS 761 A3 (Titanic) |
| t-tests, ANOVA, chi-square | MIS 761 A4 (Titanic) |
| Systems of equations, matrices | MIS 761 A5 (NumPy) |
| Multiple regression, VIF, assumptions | MIS 761 A6 (Boston housing) |
| Logistic regression, odds ratios | MIS 761 A7 (Titanic) |
| Train/test, NB vs. logistic vs. tree, overfitting | MIS 761 A8 |
| KNN, trees, bagging | MIS 761 A9 |
| K-Means, cluster-then-regress | MIS 761 A10 (housing) |
| Regression pipeline (insurance charges) | ITE 622 Assignment1_Regression |
| Classification pipeline, feature importance | ITE 622 Assignment1_Classification |
| NB vs. logistic comparison | ITE 622 Assignment2 |
| Association rules (Apriori) | Assignment3 (groceries) |
| Text classification end-to-end | Midterm Q11 |
| Model validation deep dive (stratification, leakage, learning curves) | Module 13 — extension beyond coursework |
| Data quality assessment, Colab/HuggingFace | MIS 769 HW1 |
| MapReduce, Spark, partitions | MIS 769 HW2 |
| Text preprocessing (spaCy/NLTK) | MIS 769 HW3 |
| Word2Vec embeddings | MIS 769 HW4 |
| BERTopic, LLMs, RAG, prompting, LoRA | MIS 769 HW5–HW8 |
| Python fundamentals | DA 621 (full course) |
| SQL, joins, MySQL + Python | ITE 451/651 |
| Databases & storage (NoSQL, warehouses, formats) | Module 9 — extension of ITE 451/651 |
| SVM, neural nets, TensorFlow/Keras | ITE 622 (weeks 6–11) |
