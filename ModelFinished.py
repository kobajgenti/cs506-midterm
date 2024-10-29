import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack
import time
from contextlib import contextmanager
from typing import Dict, Optional
from tqdm import tqdm
import csv
from io import StringIO

class Timer:
    """Utility class to track execution times of different code sections"""
    def __init__(self):
        self.times: Dict[str, list] = {}
        self.current_timer: Optional[str] = None
        self.start_time: Optional[float] = None

    @contextmanager
    def track(self, name: str):
        try:
            self.start_time = time.time()
            self.current_timer = name
            yield
        finally:
            if self.current_timer:
                elapsed = time.time() - self.start_time
                if name not in self.times:
                    self.times[name] = []
                self.times[name].append(elapsed)

    def summary(self):
        print("\nTiming Summary:")
        print("-" * 60)
        print(f"{'Operation':<30} {'Avg Time (s)':>10} {'Total Time (s)':>15}")
        print("-" * 60)

        total_time = 0
        for name, times in self.times.items():
            avg_time = np.mean(times)
            total = np.sum(times)
            total_time += total
            print(f"{name:<30} {avg_time:>10.3f} {total:>15.3f}")

        print("-" * 60)
        print(f"{'Total Time':<30} {'-':>10} {total_time:>15.3f}")

class MoviePredictor:
    def __init__(self, train_size=25000, random_state=42):
        self.timer = Timer()
        self.train_size = train_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(
            max_features=7500,
            min_df=5,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.user_stats = None

    def precompute_user_statistics(self, train_df, cache_file='user_stats.pkl'):
        import os
        if os.path.exists(cache_file):
            print("Loading pre-computed user statistics...")
            self.user_stats = pd.read_pickle(cache_file)
            print(f"Loaded user statistics for {len(self.user_stats)} users")
        else:
            print("Computing user statistics...")
            self.user_stats = (train_df.groupby('UserId')['Score']
                             .agg(['mean', 'count'])
                             .reset_index()
                             .rename(columns={'mean': 'user_avg_rating',
                                            'count': 'user_review_count'}))
            self.user_stats.to_pickle(cache_file)

    def create_balanced_training_sample(self, df):
        samples_per_rating = self.train_size // 5
        sampled_dfs = []
        for rating in range(1, 6):
            rating_df = df[df['Score'] == rating]
            n_samples = min(samples_per_rating, len(rating_df))
            rating_sample = rating_df.sample(n=n_samples, random_state=self.random_state)
            sampled_dfs.append(rating_sample)
        return pd.concat(sampled_dfs, ignore_index=True)

    def create_features(self, df):
        df = df.copy()
        df.loc[:, 'Summary'] = df['Summary'].fillna('').astype(str)
        df.loc[:, 'Text'] = df['Text'].fillna('').astype(str)
        df.loc[:, 'CombinedText'] = df['Summary'] + ' ' + df['Text']
        
        df.loc[:, 'UserId'] = df['UserId'].fillna('UNKNOWN')
        df = df.merge(self.user_stats, on='UserId', how='left')

        features = pd.DataFrame()
        denominator = df['HelpfulnessDenominator'].replace(0, 1)
        features['helpfulness_ratio'] = df['HelpfulnessNumerator'] / denominator
        features['review_length'] = df['Text'].str.len()
        features['summary_length'] = df['Summary'].str.len()
        features['user_avg_rating'] = df['user_avg_rating'].fillna(self.full_train_df['Score'].mean())
        features['user_review_count'] = df['user_review_count'].fillna(1)

        return features, df['CombinedText']

    def train_model(self, train_df):
        with self.timer.track("Total Training"):
            self.full_train_df = train_df
            self.precompute_user_statistics(train_df)
            
            balanced_df = self.create_balanced_training_sample(train_df)
            train_numerical, train_text = self.create_features(balanced_df)
            
            train_text_features = self.vectorizer.fit_transform(train_text)
            train_numerical_scaled = self.scaler.fit_transform(train_numerical)
            train_numerical_sparse = csr_matrix(train_numerical_scaled)
            
            X_train = hstack([train_text_features, train_numerical_sparse])
            y_train = balanced_df['Score']
            
            self.model.fit(X_train, y_train)
        
        self.timer.summary()

    def process_test_data(self, test_df, train_df):
        with self.timer.track("Total Processing"):
            test_reviews = []
            missing_ids = []
            predictions_list = []

            batch_size = max(1, int(len(test_df) * 0.004))
            num_batches = (len(test_df) + batch_size - 1) // batch_size

            for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(test_df))
                batch_df = test_df.iloc[start_idx:end_idx]

                batch_reviews = []
                for _, row in batch_df.iterrows():
                    review_data = train_df[train_df['Id'] == row['Id']]
                    if len(review_data) > 0:
                        batch_reviews.append(review_data)
                        test_reviews.append(row['Id'])
                    else:
                        missing_ids.append(row['Id'])

                if batch_reviews:
                    batch_data = pd.concat(batch_reviews, ignore_index=True)
                    numerical_features, text_features = self.create_features(batch_data)
                    text_features_transformed = self.vectorizer.transform(text_features)
                    numerical_scaled = self.scaler.transform(numerical_features)
                    numerical_sparse = csr_matrix(numerical_scaled)
                    X = hstack([text_features_transformed, numerical_sparse])
                    batch_predictions = self.model.predict(X)
                    predictions_list.extend(batch_predictions)

            if predictions_list:
                predictions = np.array(predictions_list)
                predictions = np.clip(predictions, 1, 5)
                prediction_dict = dict(zip(test_reviews, predictions))
                final_predictions = []
                mean_rating = train_df['Score'].mean()

                for test_id in test_df['Id']:
                    final_predictions.append(
                        prediction_dict.get(test_id, mean_rating)
                    )

                return np.array(final_predictions)
            return None

def round_predictions(predictions, ids, output_file):
    """
    Round the predicted scores and save them to a CSV file
    
    Args:
        predictions: numpy array of predicted scores
        ids: array-like of review IDs
        output_file: string, path to output CSV file
    """
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Id', 'Score'])  # Write header
        
        for id_num, score in zip(ids, predictions):
            # Round score to nearest integer
            rounded_score = int(round(float(score)))
            # Ensure score is between 1 and 5
            rounded_score = max(1, min(5, rounded_score))
            csv_writer.writerow([id_num, rounded_score])

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print(f"Loaded {len(train_df)} training reviews and {len(test_df)} test cases")

    # Initialize and train predictor
    predictor = MoviePredictor(train_size=25000)
    predictor.train_model(train_df)

    # Generate predictions
    predictions = predictor.process_test_data(test_df, train_df)

    if predictions is not None:
        # Generate raw predictions file
        raw_submission_df = pd.DataFrame({
            'Id': test_df['Id'],
            'Score': predictions
        })
        raw_submission_df.to_csv('raw_submission.csv', index=False)
        print(f"\nRaw predictions saved to raw_submission.csv ({len(predictions)} predictions)")
        
        # Generate rounded predictions file
        round_predictions(predictions, test_df['Id'], 'rounded_submission.csv')
        print(f"Rounded predictions saved to rounded_submission.csv ({len(predictions)} predictions)")
    else:
        print("Failed to generate predictions")

if __name__ == "__main__":
    main()