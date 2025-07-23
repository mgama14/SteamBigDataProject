# GamaSteam: Predicting Steam Review Sentiment with PySpark

Welcome to **GamaSteam**, my deep-dive project into predicting video game review sentiment using Steam’s massive reviews dataset, PySpark, and machine learning. This was my personal big data capstone—a months-long journey where I learned how to wrangle millions of rows of noisy text, engineer meaningful features, and evaluate a real predictive model. All of it is built on Google Cloud with PySpark in Jupyter Notebooks. Every decision, experiment, and mistake was part of the process.

---

##  Project Summary

This project started with a simple question: **Can we predict whether someone will leave a positive or negative review on Steam based on their behavior and metadata?**

Using nearly 50 million reviews from [this Kaggle dataset](https://www.kaggle.com/datasets/nikdavis/steam-reviews), I set out to:

- Clean and preprocess the data using PySpark  
- Engineer new features to capture player behavior  
- Train and validate a Decision Tree classifier  
- Evaluate model performance using metrics and visualizations  
- Visualize patterns like review timing and recency bias  

To kick things off, I used a Linux virtual machine on Google Cloud Platform to authenticate with Kaggle and download the full dataset directly into my storage bucket. This setup thankfully let me handle the large volume of data securely and efficiently without downloading anything to my local machine.

---

##  Exploratory Data Analysis (EDA)

Before modeling, I explored the distributions of key features like playtime, language, and review counts to check for outliers and spot trends. This helped inform how I handled missing values and which variables to keep during feature selection.

##  Project Structure (Google Cloud Bucket)

```
/gamasteamreviews
├── /landing                # Raw data (CSV, converted to Parquet)
├── /cleaned                # Cleaned and filtered dataset
├── /trusted                # Feature-engineered dataset
└── /models                 # Saved Decision Tree model
```

---

##  Data Cleaning & Preparation

The original dataset was messy—nulls, redundant columns, and strange formats. I handled this by:

- Converting from CSV to Parquet to ease Spark operations
- Selecting the most relevant columns for prediction:
  - `author_num_games_owned`
  - `author_num_reviews`
  - `author_playtime_forever`
  - `author_playtime_last_two_weeks`
  - `author_playtime_at_review`
  - `language`, `voted_up`, `steam_purchase`, `received_for_free`
- Filling missing numeric values with the **median**, and unknown categories with default labels like `"unknown"`

This resulted in a much leaner, more meaningful dataset ready for modeling.

---

##  Feature Engineering

To strengthen the predictive power of my model, I created several features:

###  `time_of_day`

Extracted from the UNIX timestamp of the review to categorize when a review was left:

- Morning (6AM–12PM)
- Afternoon (12PM–6PM)
- Evening (6PM–12AM)
- Night (12AM–6AM)

This allowed me to explore whether review sentiment changes based on when players write them.

###  `recency_bias`

Calculated the time (in days) between the last time the user played the game and when they wrote the review. Hypothesis: people who review immediately after playing may be more emotional and leave stronger feedback.

###  `played_after_review`

Boolean value that checks whether the player returned to the game *after* reviewing. This was meant to capture lasting engagement—do people keep playing a game they liked enough to praise?

All categorical variables were indexed using `StringIndexer` to prepare them for the Decision Tree model.

---

##  Model Training

I trained a **Decision Tree Classifier** using PySpark ML, splitting the dataset into 70% training and 30% testing. I used **5-fold cross-validation** and tuned parameters like:

- `maxDepth`: [5, 10, 15]
- `maxBins`: [32, 64]

The best model achieved the following metrics:

- **Accuracy**: \~0.86
- **Precision**: \~0.84
- **Recall**: \~0.91
- **F1-Score**: \~0.87
- **AUC**: 0.71

These results are surprisingly strong considering I did no textual sentiment analysis at all—only metadata and behavioral signals.

---

##  Data Visualizations

### 1. 📈 ROC Curve – Model Performance

<img width="400" height="300" alt="roc curve" src="https://github.com/user-attachments/assets/7fdf3157-c35a-4b5b-bb28-139a95c8ecf2" />


This shows the true positive rate vs. false positive rate. The model scored **AUC = 0.71**, indicating decent predictive capability.

---

### 2. 📉 Confusion Matrix – Model Predictions

<img width="400" height="300" alt="confusion matrix" src="https://github.com/user-attachments/assets/80799f27-8b18-4d8d-82d9-89224c371ef5" />

This matrix shows how often the model got reviews right or wrong. We can see a lot of correctly predicted positive reviews (\~417k), but also a high false positive count (\~66k).

---

### 3. 🌟 Feature Importance

<img width="400" height="300" alt="feature importance" src="https://github.com/user-attachments/assets/69b0346c-94c5-41f3-9c37-06a4a1e9c136" />

Unsurprisingly, `author_playtime_forever` had the most predictive power. Playtime tends to strongly correlate with enjoyment, and therefore review positivity.

`played_after_review` and `time_of_day` also showed high importance—strong proof that temporal patterns affect sentiment.

---

### 4. 🔥 Correlation Heatmap

<img width="400" height="300" alt="confusion matrix" src="https://github.com/user-attachments/assets/e20045ff-4408-4847-8648-64c536d00de0" />

This visual shows how features relate to each other. Notably:

- `author_playtime_at_review` and `author_playtime_forever` are highly correlated
- `recency_bias` is mostly independent
- `voted_up` has only weak correlations, meaning multiple subtle patterns influence sentiment

---

### 5.  Sentiment by Time of Day

#### (a) Distribution

<img width="400" height="300" alt="sentiment distribution by time" src="https://github.com/user-attachments/assets/50b039cc-ffe3-4bea-8ba7-179e10146527" />


Evening and afternoon had the highest review counts. Most reviews were positive, regardless of time.

#### (b) Proportion of Negatives

<img width="400" height="300" alt="proportion of negative reviews by time of day" src="https://github.com/user-attachments/assets/70cae0eb-a6ce-49ba-857f-49e96a064115" />


Interestingly, **morning reviews were slightly more negative** than other times.

---

### 6.  Played After Review

<img width="400" height="300" alt="proportion of sentiment by played after review" src="https://github.com/user-attachments/assets/02f5046c-606f-46b8-9bed-11ca42bcb973" />


Players who kept playing after reviewing were far more likely to have left a positive review. This suggests strong predictive value.

---

##  Model Output Storage

-  Model saved to: `/models/decision_tree_model`
-  Feature-engineered dataset saved to: `/trusted/all_reviews_with_features.parquet`

These outputs can be reused to further improve the model later, or to compare different classifiers.

---

##  Lessons Learned

- Metadata **can** predict review sentiment reasonably well, but not perfectly.
- Recency and engagement (like `played_after_review`) matter more than I expected.
- Even a basic decision tree, with useful features, can uncover meaningful patterns.
- Performance improves DRAMATICALLY with clean Parquet data.

---

##  Next Steps

If I were to continue GamaSteam, I'd probably do the following:

- Try more models (Random Forest, Gradient Boosted Trees)
- Incorporate text features from the review body
- Explore sentiment by genre or publisher
- Attempt to build a dashboard using Power BI or Streamlit

---

##  Final Thoughts

This was the first big data project where I felt like I was telling a story, not just processing numbers. Every feature came from a theory about how players behave, taking into account my own personal experiences with how I play games and my own Steam account patterns. I got to see how those theories held up, and I was able to compare my findings with my own Steam profile, which made everything feel quite a bit more rewarding.

Thanks for reading!
