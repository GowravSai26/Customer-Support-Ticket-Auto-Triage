import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import time

# Ensure stopwords are available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

print("--- Starting Model Training ---")

# 1. Load Data
df = pd.read_csv('tickets.csv')
print(f"Dataset loaded. Shape: {df.shape}")
print("Categories distribution:\n", df['Category'].value_counts())

# 2. Preprocessing
df['text'] = (df['Subject'].fillna('') + ' ' + df['Description'].fillna(''))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Category'])
X = df['cleaned_text']

print(f"Encoded categories: {list(label_encoder.classes_)}")

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Pipeline
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# 5. Evaluation
print("\n--- Model Evaluation ---")
predictions = model_pipeline.predict(X_test)

start_time = time.time()
model_pipeline.predict(X_test[:1])
end_time = time.time()
latency = (end_time - start_time) * 1000

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, target_names=label_encoder.classes_, zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Average Latency: {latency:.4f} ms per prediction")
print("Classification Report:\n", report)

# 6. Save
joblib.dump(model_pipeline, 'ticket_classifier_model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
print("\nModel and encoder saved successfully.")
print("--- Script Finished ---")
