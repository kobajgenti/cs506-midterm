# Movie Rating Analysis Project

## Overview and Methodology
This project focused on developing a machine learning system to predict movie ratings based on review text and metadata. Looking back, what started as a straightforward classification task turned into quite an adventure in feature engineering and performance optimization. Here's my journey through the development process.

## 1. Initial Data Analysis

My first step was understanding what I was working with. I started with basic data exploration:

```python
print("\nFirst Row of Training Data (with scores):")
print(train.head(1))
```

This simple print statement saved me hours later - it showed that some reviews would be missing scores (these were the ones I needed to predict). I then used `matplotlib.pyplot` to visualize the distribution of ratings. I wanted to make sure I wouldn't accidentally bias my model by oversampling certain ratings.

Next obvious step was to calculate average movie ratings - this came from the realization that some movies are just inherently better or worse than others. Implementing `movie_stats = train.groupby('ProductId').agg()` revealed an interesting pattern: while there were plenty of movies with all 5-star ratings (among those with 10+ reviews), none had all 1-star ratings. This made me wonder if there was some inherent positivity bias in online reviews.

## 2. Movie Rating Consistency Analysis

I spent quite a bit of time (probably too much in retrospect) trying to quantify rating consistency. The idea came from noticing that a 4.5-star average could mean very different things:
- Everyone genuinely rated 4-5 stars
- Half gave 5 stars, half gave 1 star

I implemented a consistency check using `CONSISTENCY_THRESHOLD = 0.25` that looked at the bottom quartile of standard deviations. My `categorize_movies()` function evolved from this idea:

```python
def categorize_movies(movie_stats):
    std_threshold = movie_stats['rating_std'].quantile(CONSISTENCY_THRESHOLD)
    
    def assign_category(row):
        is_consistent = row['rating_std'] <= std_threshold
        high_rating = row['avg_rating'] >= RATING_HIGH_THRESHOLD
        # ... more logic here
```

The results were interesting but ultimately not as useful as I hoped:
```
Working with a sample of 1,000,000 reviews
Original dataset size: 1,485,341 reviews
Sample percentage: 67.3%

Category Distribution:
Consistently Good: 7,840 movies (15.69%)
Good but Mixed: 5,752 movies (11.51%)
Mixed Reviews: 23,490 movies (47.02%)
Poor but Controversial: 4,560 movies (9.13%)
Consistently Poor: 126 movies (0.25%)
Insufficient Reviews: 8,187 movies (16.39%)
```

<img src="https://github.com/kobajgenti/cs506-midterm/blob/main/MRC.png" alt="Movie Raiting Scatter Plot" width="300"/>

After spending hours fine-tuning these categories, I ended up abandoning this approach. Too many movies fell into the "Mixed Reviews" category, making the classification less useful than I'd hoped. Still, the effort wasn't wasted - it gave me insights that helped with user feature development.

## 3. User Features

Looking at the data, I noticed the average user had 4.11 reviews. This was a goldmine - enough reviews per user to actually analyze their rating patterns! I developed two key features:

### User Bias
I wanted to identify users who consistently rated above or below average. The `calculate_user_bias()` function checked if their ratings were consistently ±0.5 std away from the mean:

```python
def calculate_user_bias(df):
    global_average = df['Score'].mean()
    movie_averages = df.groupby('ProductId')['Score'].mean()
    comparison_df['rating_bias'] = comparison_df['Score'] - comparison_df['movie_average']
```

The results were fascinating:
```
=== BIAS DISTRIBUTION ===
Mean bias: 0.037
Std deviation: 0.615
Min bias: -3.669
Max bias: 2.800

=== RATING BEHAVIOR CATEGORIES ===
Harsh critics (bias < -0.5): 21,225 users (17.1%)
Generous critics (bias > 0.5): 27,865 users (22.5%)
Neutral critics: 74,870 users (60.4%)
```

### User Credibility
This feature was kind of fun to develop. Instead of treating all reviews equally, I wrote `calculate_user_credibility()` to analyze how helpful other users found each reviewer:

```python
def calculate_user_credibility(df):
    user_stats = df.groupby('UserId').agg({
        'HelpfulnessNumerator': 'sum',
        'HelpfulnessDenominator': 'sum',
        'Score': 'count'
    })
```

The results were eye-opening:

| UserID | Credibility Score | Helpfulness Ratio | Reviews | Total Helpful Votes |
|--------|------------------|-------------------|----------|-------------------|
| A1GGOC9PVDXW7Z | 100.00 | 0.94 | 940 | 8,484 |
| A27H9DOUGY9FOS | 99.07 | 0.92 | 1,012 | 24,164 |
| ABH4G7TVI6G2T | 99.00 | 0.93 | 867 | 9,631 |

## 4. Text Analysis

For text analysis, I chose TF-IDF mainly because we had used it successfully in our BBC article analysis exercise. First step was combining the text fields:

```python
df['CombinedText'] = df['Summary'].fillna('') + ' ' + df['Text']
```

### Vocabulary Analysis
After some trial and error with different feature sizes, I settled on 7,500 words based on an elbow test. The vectorizer setup was straightforward:

```python
tfidf = TfidfVectorizer(
    max_features=7500,
    min_df=5,
    max_df=0.95,
    strip_accents='unicode',
    lowercase=True,
    stop_words='english'
)
```

### Adjective-Weighted TF-IDF
This was the "aha!" moment of the project. When I printed out the top TF-IDF terms, I noticed a problem:

```
Review text: GOOD FUN FILM While most straight to DVD films are not worth watching...

Top terms by TF-IDF score:
elvis: 0.6054
lady: 0.1795
thank: 0.1765
```

The model was focusing on nouns and context words instead of sentiment-carrying adjectives! I implemented adjective weighting using NLTK, and the improvement was immediate:

```
REVIEW #3 - RATING: 4.0 / 5

Before (Regular Top Words):
- times (NNS): 0.360
- doubt (NN): 0.258
- effort (NN): 0.252

After (Adjective-Weighted):
- great (JJ): 0.424 (adj)
- good (JJ): 0.414 (adj)
- entertaining (JJ): 0.394 (adj)
```

## 5. Model Development

### Feature Matrix
The final feature matrix combined:

1. Text features:
   - Adjective-boosted TF-IDF words
2. Numerical features:
   - Helpfulness metrics (numerator, denominator, ratio)
   - Text length features (review_length, summary_length)
   - User features (user_bias, user_credibility)

### Model Performance
Using balanced sampling (20% per rating category), the results were actually pretty good:
```
Model Performance Metrics:
Training RMSE: 0.4059
Test RMSE: 0.3757
Training MAE: 0.2006
Test MAE: 0.1437
R² Score: 0.9294

Prediction Accuracy:
Exact predictions: 88.60%
Within one star: 98.72%
Within two stars: 99.94%
```

<img src="https://github.com/kobajgenti/cs506-midterm/blob/main/CM.png" alt="Confusion Matrix 88% accuracy" width="300"/>

## Other Ideas I Had

I was doing this assignment basically from Friday 3 PM to Saturday 5 AM (had to travel Saturday because of elections) and then all day Sunday. There were so many things I wanted to try but ran out of time for:

### Investigate Time Feature
I wanted to check if reviews follow temporal patterns. For instance:
- Do horror movies get better ratings during Halloween?
- Are Christmas movies rated differently during holidays?
- Have rating patterns changed over the years?
- Do late-night reviews tend to be more extreme?

### Sentiment Analysis 
I started but couldn't finish a more sophisticated sentiment analysis. The key idea was to handle negations properly:
- Example: "bad" (Negative) vs "not bad" (Positive)

I wrote this function to start automating sentiment labeling:
```python
def label_sentiment(text, rating):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    # Never got to finish this...
```

## Adapting to Full Dataset

After getting great results on the balanced sample, scaling up to the full dataset was... humbling. My Google Colab kept crashing even after buying more RAM. I added timing code to investigate:

```
Feature Creation Breakdown:
Data copy took: 0.003s
Text preprocessing took: 0.004s
User features merging took: 0.058s
Helpfulness ratio calculation took: 0.003s
Text length calculations took: 0.003s
User feature filling took: 0.011s
Feature creation took: 0.083s
Text vectorization took: 0.110s
Numerical scaling took: 0.003s
Feature stacking took: 0.001s
Processing batches:   0%|          | 0/250 [00:00<?, ?it/s]
```

The user statistics calculation was killing performance:
```
User statistics calculation took: 0.830s
- MAJOR BOTTLENECK
- Calculates average ratings and review counts for each user
- Slow because it's grouping and aggregating the entire training dataset each time
```

I tried everything:
1. Pre-computing statistics
2. Batch processing
3. Moving to local machine
4. Adding more RAM
5. Optimizing database queries

In the end, I barely got it working in time for submission. I got second worst results in class. ☹️

## Learning Experience

Even though my model performed horribly, I honestly enjoyed the experience. Key takeaways:
1. Start with performance testing EARLY
2. Test scaling issues before spending days on feature engineering
3. Sometimes simple features with good scaling are better than perfect features that won't run
4. Sleep is important - some of my best ideas came after taking breaks
5. Keep a development diary - writing this report was much easier because I had notes

For next time:
- Will start with small-scale tests for computational complexity
- Plan to set aside 25% of time for optimization
- Will implement proper progress tracking from day 1
- Need to learn more about distributed computing

---
Author: Koba Jgenti
Date: October 28, 2024
Location: Brookline
