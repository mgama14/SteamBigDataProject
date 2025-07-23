# GamaSteam: Predicting Steam Review Sentiment with PySpark

Welcome to **GamaSteam**, a personal capstone where I explored how player behavior correlates with review sentiment on Steam, the largest PC gaming platform in the world. As someone deeply passionate about video games, I wanted to combine that love with data science, and this project was the result of months of hard work with PySpark, machine learning, and Google Cloud.

Using nearly 100 million user reviews from a Kaggle dataset, I built an end-to-end machine learning pipeline: cleaning raw data, engineering new behavioral features, training a Decision Tree classifier, and visualizing the results. Everything is powered through a Linux virtual machine on Google Cloud, with Jupyter Notebook as the development environment.

This was my chance to see how well we can predict user sentiment based *only* on metadata and play behavior—no text sentiment analysis, just pure patterns. An in-depth PDF writeup is also included in this repository for a deeper dive.

---

##  Project Summary

This project started with a simple question: **Can I predict whether someone will leave a positive or negative review on Steam based on their behavior and metadata?**

Using nearly 100 million reviews from [this Kaggle dataset](https://www.kaggle.com/datasets/kieranpoc/steam-reviews), I set out to:

- Clean and preprocess the data using PySpark  
- Engineer new features to capture player behavior  
- Train and validate a Decision Tree classifier  
- Evaluate model performance using metrics and visualizations  
- Visualize patterns like review timing and recency bias  

To kick things off, I used a Linux VM on Google Cloud to authenticate with Kaggle and download the full dataset directly into my storage bucket. This setup thankfully let me handle the large volume of data securely and efficiently without downloading anything to my local machine.

---
##  Project Structure (in my Google Cloud Bucket)

```
/gamasteamreviews
├── /landing                # Raw data (CSV, converted to Parquet)
├── /cleaned                # Cleaned and filtered dataset
├── /trusted                # Feature-engineered dataset
└── /models                 # Saved Decision Tree model
```

---

##  Exploratory Data Analysis (EDA)

In the EDA portion of this project, I used PySpark to gain a foundational exploratory look at the dataset. Holistically, the dataset comprises a substantial 113,883,717 records. A quick statistical overview revealed some fun outliers: the user with the most Steam games owned had 33,345 games, and the longest recorded playtime clocked in at 97,317 hours—that’s over 4,000 days.

To better understand the dataset, I created a few very simple graphs.

### Review Count by Language (Top 5)

<img width="400" height="300" alt="reviewbylanguage" src="https://github.com/user-attachments/assets/ad2f625e-0dd0-403c-845f-e3e938880de8" />

Unsurprisingly, the overwhelming majority of Steam reviews are in English, followed by Simplified Chinese, Russian, Spanish, and Brazilian Portuguese. I found it interesting that Chinese came in second, despite China's regulatory limitations with Steam. It’s a good reminder that this platform has a huge global user base.

### Playtime vs. Review Count

<img width="400" height="300" alt="hoursvreviews" src="https://github.com/user-attachments/assets/07bf5a93-1aa4-4d9f-8824-fd506880f5d6" />

There’s a noticeable cluster of users who left reviews despite having little to no playtime. These reviews can stem from things like game crashes, installation issues, or review bombing, so they might not reflect real gameplay sentiment. As I moved into modeling, I kept this in mind and made sure to treat playtime-related features thoughtfully.

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

This stage was all about finding ways to capture player behavior and emotional patterns without ever touching the text of the reviews.
To strengthen the predictive power of my model, I engineered **three new features** based on timestamps and behavioral cues:

###  `time_of_day`

Using the review creation timestamp, I binned each review into one of four categories:

- Morning (6AM–12PM)
- Afternoon (12PM–6PM)
- Evening (6PM–12AM)
- Night (12AM–6AM)

The idea being that players might express different moods or sentiments depending on the time of day.

###  `recency_bias`

This feature calculates the number of days between when the user last played the game and when they wrote the review. The closer the review is to the last play session, the more likely it is that the sentiment reflects a fresh emotional response, whether good or bad.

This idea came from my own experience as a gamer: I tend to rate things more generously when they're still fresh in my mind, as excitement usually overpowers any of my bigger criticisms with the game... unless it's **really** bad. 

###  `played_after_review`

This is a boolean feature that flags whether the user played the game again after submitting their review. The idea is simple: if someone keeps playing, chances are they actually enjoyed the game (or are a masochist), even if the review didn’t say it outright.

Alongside those, I retained and prepared several existing columns that seemed valuable:

| Feature | Treatment |
|--------|-----------|
| `author_num_games_owned` | Used as-is |
| `author_num_reviews` | Used as-is |
| `author_playtime_forever` | Used as-is |
| `author_playtime_last_two_weeks` | Used as-is |
| `author_playtime_at_review` | Used as-is |
| `author_last_played` | Used as-is |
| `recency_bias` | Used as-is |
| `played_after_review` | Used as-is |
| `language` | Encoded with StringIndexer |
| `time_of_day` | Encoded with StringIndexer |
| `timestamp_created` | Used as-is |

I handled missing values in the newly engineered columns using **median imputation**, and encoded all categorical features with a `StringIndexer`. Finally, I bundled everything into a single vector using PySpark's `VectorAssembler`.

---

##  Model Training

With all features ready, I trained a **Decision Tree Classifier** using PySpark ML. I split the data into **70% training** and **30% testing**, then used **5-fold cross-validation** to tune hyperparameters:

- `maxDepth`: [5, 10, 15]
- `maxBins`: [32, 64]

The best model was evaluated using multiple metrics:

| Metric | Value |
|--------|-------|
| **Cross-Validated Accuracy** | 85.61% |
| **Precision** | 82.45% |
| **Recall** | 85.61% |
| **F1-Score** | 80.91% |

These results are honestly stronger than I expected. A recall of 85.61% tells me the model is great at detecting actual positive reviews, while the precision of 82.45% shows it made a few false positives—but not many. The F1-score reflects a healthy balance between the two.

---

##  Data Visualizations

To get a deeper understanding of the patterns influencing sentiment, I created several visualizations that explore the behavioral signals behind each review.

---

### 🕒 1. Sentiment by Time of Day

#### (a) Review Distribution

<img width="400" height="300" alt="sentiment distribution by time" src="https://github.com/user-attachments/assets/50b039cc-ffe3-4bea-8ba7-179e10146527" />

Positive reviews dominated across all time periods, but **afternoons and evenings** saw the highest volume of reviews overall. Mornings and nights trailed behind—nights especially, which was a bit surprising. It seems players are most active (and most expressive) in the later parts of the day.

#### (b) Proportion of Negatives

<img width="400" height="300" alt="proportion of negative reviews by time of day" src="https://github.com/user-attachments/assets/70cae0eb-a6ce-49ba-857f-49e96a064115" />

While the difference was small, **morning reviews had the highest proportion of negative sentiment**, and **evening reviews had the lowest**. This again suggests that timing might not only influence *when* players leave reviews—but *how* they feel when they do. And that they tend to be much more generous later in the day!

---

### 🎮 2. Played After Review

<img width="400" height="300" alt="proportion of sentiment by played after review" src="https://github.com/user-attachments/assets/02f5046c-606f-46b8-9bed-11ca42bcb973" />

This one was clear: players who **kept playing after reviewing** were far more likely to leave positive feedback (and it also clearly illustrates the aforementioned masochists). It reinforces the idea that continued engagement is a powerful signal of satisfaction, while those who drop off are more likely to express disappointment.

---

### 🔥 3. Correlation Heatmap

<img width="400" height="300" alt="confusion matrix" src="https://github.com/user-attachments/assets/e20045ff-4408-4847-8648-64c536d00de0" />

This heatmap shows how features relate to one another. Some interesting takeaways:

- **`author_playtime_forever`** and **`author_playtime_at_review`** were tightly linked (correlation: 0.78), showing they capture similar but distinct signals.
- **`author_playtime_last_two_weeks`** only moderately correlated with total playtime (0.33), which reflects the difference between long-term investment and recent interest.
- **`recency_bias`** was largely independent, making it a unique contribution to the model.
- **`author_num_reviews`** and **`author_num_games_owned`** were weakly linked (0.31), suggesting that more invested users tend to leave more feedback.

Interestingly, no feature had a particularly strong correlation with `voted_up`, which supports the idea that sentiment arises from a nuanced mix of factors—not any single one.

---

### 🧠 4. Feature Importance (Decision Tree)

<img width="400" height="300" alt="feature importance" src="https://github.com/user-attachments/assets/69b0346c-94c5-41f3-9c37-06a4a1e9c136" />

The decision tree model revealed that **`author_playtime_forever`** was the most influential feature when predicting review sentiment. Not surprising; more time spent usually means more enjoyment (masochists excluded).

Close behind was **`played_after_review`**, which echoed the pattern seen in our earlier visualization: players still engaged with the game tend to leave positive feedback.

Other helpful contributors included **`time_of_day_indexed`**, **`recency_bias`**, and **`author_num_reviews`**, all offering subtle but meaningful behavioral insights.

---

### 📉 5. Confusion Matrix – Model Predictions

<img width="400" height="300" alt="confusion matrix" src="https://github.com/user-attachments/assets/80799f27-8b18-4d8d-82d9-89224c371ef5" />

This matrix shows how often the model predicted review sentiment correctly.

- **True Positives**: ~417,000
- **False Positives**: ~66,000

While the model was solid at predicting positive reviews, it struggled a bit more with negatives—often mistaking them for positives. This could mean the model is picking up on engagement patterns but missing some of the more subtle signs of dissatisfaction.

---

### 📈 6. ROC Curve – Model Performance

<img width="400" height="300" alt="roc curve" src="https://github.com/user-attachments/assets/7fdf3157-c35a-4b5b-bb28-139a95c8ecf2" />

The ROC curve gives us a broader view of the model’s performance. With an **AUC of 0.71**, the model performs significantly better than random guessing, though there's room to grow. It’s a decent baseline!

---

##  Final Thoughts

Wrapping up this project, I feel like I finally got to do something that combined my love of gaming with everything I’ve learned. From the very beginning, I wanted to figure out what really influences how people feel about the games they play. It's not random; it’s deeply tied to behavior, timing, and how someone experiences a game in the moment.

Some of the most important takeaways came from engagement metrics. Total playtime was huge, but what really stood out was whether players kept playing after leaving a review. If someone keeps coming back, they’re probably enjoying themselves. Timing played a role too—afternoon and evening reviews were the most positive, and reviews written soon after playing tended to be more intense, whether good or bad. That really drove home how much emotional context matters.

The model itself did pretty well. A cross-validated accuracy of 85.61% and an AUC of 0.71 isn’t perfect, but it’s solid for my first attempt at creating a model like this. It struggled a bit with catching negative reviews, but overall, it got the sentiment right more often than not.

Beyond the stats, what I enjoyed most was seeing real patterns in how people play and respond to games. It made me think about my own habits. I’m definitely going to think about that next time I hit the “thumbs up” button on Steam.

This project was a mix of challenges, rabbit holes, and discoveries, and I’m really proud of how it turned out. At first, the dataset felt overwhelming, but step by step I built something that actually tells a story. 


---

### 🔮 If I Keep Going…

If I decide to expand on this project later, here’s what I’d like to try:

- Test other models like Random Forest or Gradient Boosted Trees
- Bring in text data from the reviews to do real sentiment analysis
- Look at trends by genre or publisher—like, are RPGs reviewed more kindly than shooters?
- Build a dashboard in Power BI or Streamlit to make the results interactive

There’s a lot more I could do with this dataset, but for now, I’m happy with where this ended up. Thanks for checking out my project.
