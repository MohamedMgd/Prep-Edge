# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load your dataset (use only 10,000 rows to avoid memory issues)
df = pd.read_csv("final_Merged_Questions_Consolidated.csv", nrows=10000)

# Fill missing values and ensure strings
df[['question text', 'option 1', 'option 2', 'option 3', 'option 4']] = df[
    ['question text', 'option 1', 'option 2', 'option 3', 'option 4']
].fillna("").astype(str)

# Combine question and options into one feature
df['full_text'] = (
    df['question text'] + " " + df['option 1'] + " " +
    df['option 2'] + " " + df['option 3'] + " " + df['option 4']
)

# Encode difficulty levels
label_encoder = LabelEncoder()
df['difficulty_encoded'] = label_encoder.fit_transform(df['difficulty level'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['full_text'], df['difficulty_encoded'], test_size=0.2, random_state=42
)

# Create and train the model pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)

# Decode integer labels to original text labels
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Print accuracy
print("🔍 Accuracy:", accuracy_score(y_test, y_pred))

# Print detailed classification report
print("\n📋 Classification Report:\n", classification_report(y_test_labels, y_pred_labels))

# Show confusion matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("📊 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Save model and encoder
joblib.dump(pipeline, "question_difficulty_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n✅ Model and label encoder saved successfully.")
