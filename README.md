# 6600_project_financial_analysis

# Project: Neural Network-based Option Pricing

This project explores how a neural network can be used to estimate the price of equity options. Our goal is to train a model that takes standard option parameters as input and outputs the estimated price, based on real market data.

## 🔍 Objective

Build a regression model that predicts the mid-market price of an option using features like:
- Strike price
- Option type (Call or Put)
- Time to expiration
- Implied volatility
- Underlying stock price

## 🗂️ Dataset

We’re using a sample of OptionMetrics data (from WRDS), which includes:
- `date`, `exdate`, `strike_price`, `cp_flag`
- `best_bid`, `best_offer`
- `impl_volatility`, `delta`, `gamma`, etc.
- Underlying symbol: e.g., AAPL

The option price label is calculated as:
**option_price = (best_bid + best_offer) / 2**

We use `yfinance` to get the underlying stock price on each option’s trading date.

## 🧠 Initial Model Plan

- Simple feedforward neural network (MLP)
- Input features: underlying price, strike price, implied volatility, days to expiry, call/put flag
- Output: estimated option price
- Loss: Mean Squared Error (MSE)

