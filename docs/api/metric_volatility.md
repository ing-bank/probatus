# Metric Volatility

The aim of this module is the analysis of how well a model performs on a given dataset, and how stable the performance is.

The following features are implemented:

- [TrainTestVolatility][probatus.metric_volatility.volatility.TrainTestVolatility]: Estimation of the volatility of metrics. The estimation is done by splitting the data into train and test multiple times and training and scoring a model based on these metrics.
- [SplitSeedVolatility][probatus.metric_volatility.volatility.SplitSeedVolatility]: Estimates the volatility of metrics based on splitting the data into train and test sets multiple times randomly, each time with a different seed.
- [BootstrappedVolatility][probatus.metric_volatility.volatility.BootstrappedVolatility]: Estimates the volatility of metrics based on splitting the data into train and test with static seed, and bootstrapping the train and test set.


::: probatus.metric_volatility.volatility