# Linear Regression Comparison: Scikit-learn vs PyTorch

## Overview

This project compares the performance of linear regression models implemented using Scikit-learn and PyTorch on the California Housing dataset. The goal is to explore the differences in model training, optimization, and prediction between the two popular machine learning libraries.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Matplotlib

## Data Preparation

The California Housing dataset is used, which contains information about housing prices in various regions of California. The dataset is loaded using Scikit-learn's `fetch_california_housing` function. The data is then split into training and testing sets using an 80-20 split. StandardScaler from Scikit-learn is applied to standardize the features.

## Models

Two linear regression models are implemented: one using Scikit-learn's `LinearRegression` and the other using PyTorch's `nn.Linear` module. The PyTorch model is defined as a custom class inheriting from `nn.Module`.

## Training

The Scikit-learn model is trained using the ordinary least squares (OLS) algorithm. The PyTorch model is trained using stochastic gradient descent (SGD) with a learning rate of 0.01 for 1000 epochs. The training process includes forward and backward passes, weight updates, and loss computation.

## Evaluation

Mean squared error (MSE) is used as the evaluation metric for both models. The MSE values are calculated on the test set and printed for comparison.

## Visualization

A scatter plot is created to visualize the true values of the target variable against the predicted values obtained from both models. A perfect prediction line (y = x) is also plotted for reference.

## Conclusion

This project demonstrates the implementation of linear regression using Scikit-learn and PyTorch. The comparison highlights the differences in model training, optimization techniques, and prediction results. The visualization provides insights into how well the models perform in terms of predicting housing prices.
