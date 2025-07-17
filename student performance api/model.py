import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. تحميل البيانات
df = pd.read_csv("C:\\Users\\pc\\Downloads\\student performance api\\student_performance_dataset_realistic.csv")

# 2. تقسيم المدخلات والمخرجات
X = df[['AvgPreviousScore', 'AvgTimeSpent', 'NumQuizzesTaken', 'LastScore']]
y = df['ExpectedScore']

# 3. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. إنشاء النموذج وتدريبه
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. تقييم الأداء
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📈 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📈 R2 Score: {r2:.2f}")

# 6. حفظ الموديل
joblib.dump(model, 'student_performance_model.pkl')
print("✅ Model saved as 'student_performance_model.pkl'")
