# 6600_project_financial_analysis

## Title: Option Implied Volatility Prediction Based on Recurrent Neural Network

### Introduction

This project aims to use deep learning methods to build a model to predict the implied volatility (IV) of Nasdaq 100 Index (NDX) options. We used three structures: basic recurrent neural network (RNN), long short-term memory network (LSTM) and gated recurrent unit (GRU), and compared the performance of different architectures in financial time series modeling tasks to explore their application potential in implied volatility modeling.

### Objective

- Input basic options features, such as:
- Strike Price
- Option Type (Call / Put)
- Days to Expiry
- Implied Volatility
- Mid-price
- Build a regression model and output the predicted option implied volatility.
- Compare the differences in prediction accuracy, stability and convergence speed among the three models of RNN, LSTM and GRU.

### Dataset Description

- Data source: NASDAQ-100 index options historical trading data.
- Time span: February 1, 2023 to March 31, 2023.
- Data size: 834,780 records, 19 feature fields.
- Main features include:
- Strike Price, Call/Put Flag, Days to Expiry, Best Bid, Best Offer, Delta, Gamma, Vega, Theta, Implied Volatility, etc.
- Label definition:
- Mid-price = (Best Bid + Best Offer) / 2, used to more accurately reflect the market consensus price.

### Data preprocessing

- Calculate the mid-price.
- Divide the execution price by 1000 for scale normalization.
- Numerically encode the Call/Put type (Call=1, Put=0).
- Delete records with missing key fields.
- Use StandardScaler to standardize all features (mean is 0, variance is 1).
- Use GroupKFold (grouped by transaction date) to partition the data to strictly prevent time leakage.

### Neural network model design

- Unified model structure:
- Input layer
- 1024-unit recurrent layer (RNN/LSTM/GRU, activation function is tanh)
- Dropout layer (10% dropout rate, to prevent overfitting)
- Two fully connected layers (512 units → 256 units, both using ReLU activation)
- Output layer (linear activation, regression prediction of implied volatility)
- Optimization method:
- Adam optimizer (initial learning rate 0.001)
- EarlyStopping mechanism (early termination of training to prevent overfitting)
- ReduceLROnPlateau dynamic learning rate adjustment (reducing the learning rate when performance stagnates)
- Evaluation method:
- GroupKFold 3-fold cross validation, evaluating MSE and R² indicators.

### Experimental results

- The LSTM model performs best in terms of stability and training effect.
- GRU is slightly better than RNN in terms of convergence speed and overall error control.
- All models capture the actual implied volatility trend well.
- The prediction results are highly consistent with the real data trend during sudden market events (such as the Silicon Valley Bank crisis).

### Future work prospects

- Introduce more external market indicators, such as S&P500, VIX index, Treasury yield curve, etc., to enrich input features.
- Explore the application potential of Transformer and its variants in financial time series modeling.
- Expand data coverage and introduce option trading data from multiple years and multiple market states.
- Design a model framework that supports online learning to improve the ability to respond to emergencies in real time.
