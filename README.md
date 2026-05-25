# Haley Yahiku

**B.S. Project Management · M.S. Data Analytics** &nbsp;|&nbsp; Honolulu, HI

Data Analytics professional with a background in project management, data engineering, and machine learning. Experienced in transforming complex datasets into interpretable insights and building accessible data products. Prior experience spans construction project management, elevator installation, and laboratory operations — bringing cross-industry analytical perspective to data science work.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/haley-yahiku)

---

## Skills

**Languages & Libraries**
`Python` `SQL` `pandas` `numpy` `scikit-learn` `matplotlib` `seaborn` `plotly` `scipy`

**Machine Learning & AI**
`K-Means Clustering` `Linear Regression` `NLP` `sentence-transformers` `TF-IDF` `Apriori` `Association Rules` `RAG-Inspired Retrieval` `Hypothesis Testing`

**Data Engineering**
`Data Cleaning & Preparation` `Feature Engineering` `StandardScaler` `Outlier Detection (IQR)` `Dummy Encoding` `CSV Pipeline Engineering` `Relational Databases`

**Analytics & Statistics**
`Exploratory Data Analysis` `T-Test` `ANOVA` `Correlation Analysis` `Descriptive Statistics` `Time Series Forecasting` `Financial Modeling`

**Tools & Platforms**
`Jupyter Notebook` `Google Colab` `PySpark` `Microsoft Suite` `SharePoint` `Hugging Face Datasets` `Kaggle` `GitHub Pages`

---

## Education

| Degree | Field | Institution |
|---|---|---|
| Master of Science | Data Analytics | University of Nevada, Las Vegas |
| Bachelor of Applied Science | Project Management | College of Southern Nevada |
| Associate of Science | General Studies | College of Southern Nevada |

---

## Projects

---

### 🌎 Climate Profile Segmentation & Retrieval
> Unsupervised ML · NLP · RAG-Inspired Retrieval · Brazil Municipal Data

K-Means clustering segments 5,500+ Brazilian municipalities into four distinct climate profiles based on temperature, humidity, precipitation, and solar radiation. A hybrid retrieval system allows natural language queries — combining structured filtering with semantic ranking to outperform semantic-only approaches.

| | |
|---|---|
| **Dataset** | ~3.5M weekly climate observations · Hugging Face |
| **Key Result** | 4 climate profiles identified · hybrid retrieval significantly outperformed semantic-only baseline |

**Top Stack:** `K-Means` `sentence-transformers` `StandardScaler` `cosine similarity` `plotly`

📄 [View Project Page](https://hyahiku.github.io/climate-segmentation-RAG%20system/climate-profile-segmentation.html) &nbsp;|&nbsp; 📓 [View Notebook](./climate-segmentation-RAG%20system/CODE.ipynb)

---

### 🛒 Amazon Beauty Review Complaint Analysis
> NLP · K-Means Clustering · E-Commerce · Text Classification

NLP preprocessing and K-Means clustering surface and quantify the primary drivers of customer dissatisfaction across 700K+ Amazon beauty reviews. Quality, color, and performance issues account for over 64% of all complaint drivers — providing actionable signals for product development and listing optimization.

| | |
|---|---|
| **Dataset** | 701,528 Amazon beauty reviews · Hugging Face |
| **Key Result** | 6 quantified complaint categories · Quality Issues = 26.4% of all complaints |

**Top Stack:** `NLP` `NLTK` `spaCy` `K-Means` `TF-IDF` `PySpark` `wordcloud`

📄 [View Project Page](https://hyahiku.github.io/Amazon-reviews-NLP-sentiment/amazon-beauty-nlp.html) &nbsp;|&nbsp; 📓 [View Notebook](#)

---

### 🧺 Online Retail Market Basket Analysis
> Association Rules · TF-IDF · K-Means · Retail Analytics

Apriori algorithm mines 18,536 transaction baskets for 275 association rules. Quantity bins extend co-occurrence analysis to capture purchase intensity. TF-IDF + K-Means clusters products into 5 thematic categories by item description linguistics.

| | |
|---|---|
| **Dataset** | ~400K transactions · UCI Machine Learning Repository (Dec 2010 – Dec 2011) |
| **Key Result** | 275 association rules · pink/green plate co-occurrence lift = 61.9× · 5 product clusters |

**Top Stack:** `Apriori (mlxtend)` `TfidfVectorizer` `K-Means` `association rules` `PySpark`

📄 [View Project Page](https://hyahiku.github.io/Retail-Association/online-retail-analysis.html) &nbsp;|&nbsp; 📓 [View Notebook](./Retail-Association/776_Group_Project_Rev_1.ipynb)

---

### 💰 Income & Asset Value Risk Behavior Analysis
> Hypothesis Testing · T-Test · ANOVA · Financial Analytics

Investigates whether income level is a statistically significant predictor of asset accumulation using independent t-test and one-way ANOVA across binary and quartile income groupings. Both tests reject the null hypothesis with p-values well below 0.05.

| | |
|---|---|
| **Dataset** | Risk Behavior Features · Kaggle |
| **Key Result** | t = 2.894, p = 0.005 · F = 4.833, p = 0.0041 · H₀ rejected in both tests |

**Top Stack:** `scipy.stats` `t-test` `one-way ANOVA` `EDA` `seaborn` `matplotlib`

📄 [View Project Page](https://hyahiku.github.io/Financial-Behavior-Risk/financial-behavior-risk.html) &nbsp;|&nbsp; 📓 [View Notebook](./Financial-Behavior-Risk/MIS_761_Final_Group_Project.ipynb)

---

### 🐟 Global Aquaculture Imports & Exports
> Data Engineering · Linear Regression · Forecasting · Interactive CLI

Engineers a modular analytics pipeline across 12 regional fishery CSVs. Eight dedicated functions assemble per-country trade profiles covering imports, exports, revenue, and year-over-year performance. Linear regression forecasts 5-year revenue growth via an interactive CLI.

| | |
|---|---|
| **Dataset** | 12 CSV files · 6 global regions · 2000–2015 · FAO Fishery Data |
| **Key Result** | Per-country trade profiles on demand · 2016–2020 revenue growth forecast |

**Top Stack:** `pandas` `numpy` `np.polyfit` `matplotlib` `CSV pipeline engineering` `CLI`

📄 [View Project Page](https://hyahiku.github.io/Aquatic-Imports-Exports/aquatic-imports-exports.html) &nbsp;|&nbsp; 📓 [View Notebook](./Aquatic-Imports-Exports/DA622_Final_Project.py)

---

### 🏠 Airbnb Exploratory Analysis
> EDA · Linear Regression · Hypothesis-Driven Analysis · NYC Listings

Investigates whether host flexibility (cancellation policy, instant booking, minimum nights) correlates with rental popularity across 99,729 cleaned NYC Airbnb listings. Side-by-side comparison of 5-star vs. non 5-star renters. Surprising null result — flexibility shows no meaningful correlation with rating.

| | |
|---|---|
| **Dataset** | 102,599 NYC Airbnb listings · insideairbnb.com |
| **Key Result** | No correlation between flexibility and rating · strong price/service fee linear relationship identified |

**Top Stack:** `pandas` `seaborn` `matplotlib` `linear regression` `IQR outlier detection` `EDA`

📄 [View Project Page](https://hyahiku.github.io/Airbnb-Market%20Analysis/airbnb-exploratory-analysis.html)

---

<sub>Last updated: May 2026</sub>
