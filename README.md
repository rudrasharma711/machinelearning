# 📈 S&P 500 Movement Prediction using Random Forest Classifier

This project analyzes historical data from the S&P 500 Index (`^GSPC`) and uses machine learning to predict whether the index will go up or down the next day. It leverages Yahoo Finance data via the `yfinance` library and applies a `RandomForestClassifier` from `scikit-learn` to perform the classification.

---

## 🔍 Objective

To train a binary classification model that predicts whether the closing price of the S&P 500 the next day will be higher (`1`) or not (`0`) compared to today's close.

---

## 🛠️ Tools & Libraries Used

- `yfinance`: For downloading historical S&P 500 data  
- `pandas`: Data manipulation and analysis  
- `matplotlib`: Data visualization  
- `scikit-learn`: For model building and evaluation  
- `Jupyter Notebook` / `Google Colab`: Development environment

---

## 📊 Data Processing

1. **Load data** using `yfinance`:
   ```python
   sp500 = yf.Ticker("^GSPC").history(period="max")
   ```

2. **Feature Engineering**:
   - `Tomorrow`: Shifted close value
   - `Target`: Binary column for classification (1 if tomorrow's close > today's, else 0)

3. **Additional Features** (based on moving averages):
   - `Close_Ratio_[horizon]`: Ratio of today's close to the average over a period
   - `Trend_[horizon]`: Sum of upward movements in the past `[horizon]` days  
   Horizons used: `[2, 5, 60, 250, 1000]`

---

## 🤖 Model Training

### Baseline Model:
```python
RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
```
- **Input features**: `["Close", "Volume", "Open", "High", "Low"]`
- **Precision**: ~0.53

### Enhanced Model:
```python
RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
```
- **Input features**: Rolling average and trend features
- **Custom prediction threshold**: 0.6 (for increased confidence)
- **Precision**: ~0.54

---

## 🔁 Backtesting

A custom backtesting function simulates how the model would perform over time using rolling windows:
```python
def backtest(data, model, predictors, start=2500, step=250):
    ...
```

---

## 📉 Results Summary

| Metric | Value |
|--------|-------|
| Predictions (Up) | ~1047 |
| Predictions (Down) | ~5412 |
| Final Precision | ~0.54 |
| Proportion of Up Days | ~53.7% |

The model slightly outperforms naive guessing by identifying trends over multiple time horizons.

---

## 📌 Next Steps

- Tune hyperparameters further for improved precision
- Try different models (e.g., XGBoost, LSTM)
- Add technical indicators (MACD, RSI)
- Incorporate sentiment analysis from financial news or social media

---

## 📁 File Structure

```
.
├── sp500_prediction.ipynb   # Main notebook
├── README.md                # Project overview
```

---

## ✅ Requirements

Install all dependencies using pip:

```bash
pip install yfinance pandas matplotlib scikit-learn
```

---

## 📬 Contact

Feel free to reach out with suggestions or improvements!
