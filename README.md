# 🚀 Traffic Flow Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24%2B-orange) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-brightgreen) ![Colab](https://img.shields.io/badge/Open%20In-Google%20Colab-yellow)

## 📌 Overview
This project demonstrates a machine learning model to predict travel time based on traffic flow data. It utilizes a **Random Forest Regressor** to estimate travel time based on features like speed, confidence level, and road length.

## 🎯 Features
- 📊 Generates a **synthetic dataset** for model training.
- 🤖 Trains a **Random Forest Regressor** to predict travel time.
- 📉 Evaluates the model using **Mean Absolute Error (MAE)**.
- 🔮 Predicts travel time for new traffic data.
- 📝 **Easy to use & extendable for real-world applications!**

---

## 📂 Project Structure
| File | Description |
|------|------------|
| `traffic_flow_prediction.py` | Main script to generate data, train the model, and make predictions. |
| `README.md` | Documentation for the project. |
| `requirements.txt` | List of dependencies. |

---

## 🛠 Installation & Setup
### Prerequisites
Ensure you have **Python 3.8+** and the required libraries installed.

```bash
pip install -r requirements.txt
```

Alternatively, install manually:
```bash
pip install numpy pandas scikit-learn requests
```

---

## 🚀 Running the Project
### 1️⃣ Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 2️⃣ Run Locally
```bash
python traffic_flow_prediction.py
```

---

## 🔬 How It Works
### 📌 Step 1: Generate Sample Dataset
```python
np.random.seed(42)
data = {
    "current_speed": np.random.randint(10, 100, 100),
    "free_flow_speed": np.random.randint(20, 120, 100),
    "confidence": np.random.uniform(0.5, 1.0, 100),
    "road_length": np.random.randint(1, 20, 100),
    "travel_time": np.random.randint(5, 50, 100),
}
df = pd.DataFrame(data)
```

### 📌 Step 2: Train the ML Model
```python
X = df.drop("travel_time", axis=1)
y = df["travel_time"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 📌 Step 3: Model Evaluation
```python
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
```
📊 **Model Performance:**  
✅ Mean Absolute Error: **7.21 minutes**

### 📌 Step 4: Predict New Travel Time
```python
new_data = pd.DataFrame({
    "current_speed": [50],
    "free_flow_speed": [70],
    "confidence": [0.9],
    "road_length": [10],
})
predicted_time = model.predict(new_data)
print(f"Predicted Travel Time: {predicted_time[0]} minutes")
```

🛣️ **Predicted Travel Time:** **30.05 minutes**

---

## 📊 Visualizations
### Feature Importance
```python
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Feature Importance")
plt.show()
```
**🔍 Feature Importance Graph:**
![Feature Importance](https://via.placeholder.com/600x300?text=Feature+Importance+Graph)

---

## 💾 Saving the Model
```python
import joblib
joblib.dump(model, "travel_time_model.pkl")
```

To load the model later:
```python
loaded_model = joblib.load("travel_time_model.pkl")
```

---

## 🎯 Future Enhancements
- 📡 Integrate **real-world traffic API** for dynamic predictions.
- 📌 Deploy the model using **Flask** or **FastAPI**.
- 📈 Improve accuracy using **hyperparameter tuning**.
- 📊 Add real-time **interactive dashboards**.

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License
This project is **MIT Licensed**. Feel free to use and modify.

---

## ⭐ Acknowledgments
Special thanks to the **Open Source Community** and **Machine Learning Enthusiasts**! 🚀💡

💡 **If you found this useful, please ⭐ the repo!**
