<h1 align="center">FinAutoML (Financial Forecasting with time series data)</h1>

<p align="center">
  The project aim is to cover entire spectrum of machine learning algorithmic solutions for time series forecasting in a one-stop solution by developed an python package to perform automated statistical tests and finanial forecasting experiments with 
  reporting on financial time series data.
</p>

<div align="center">
  <img src="./logo.png" alt="FinAutoMLLOGO" width="200">
</div>

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)


## About

This package is designed to aid novice and experienced professionals to play with financial time series data in-terms of both analysing the data
by statistical tests and also to experiment different forecasting algorithms and generate a detailed report for both the processes
## Installation

Download the given zip folder into your local machine. Unzip and navigate inside the folder using terminal or any console to identify the requirments.txt file. This file constains all the dependencies to install the finAutoML package. Get all the dependencies using the command,
```commandline
pip install -r ./path/requirements.txt
```

when "FinAutoML" hosted on PyPI use the following command to install the package

```commandline
pip install finautoml
```

## Usage 
```
from FinAutoML.forecast_experiment import Forecast
from FinAutoML.data_analysis import StatisticalTests

# initialise class object for statistical test for ITC stock data from yfinance
test = StatisticalTests(stock_code='ITC.NS', startdate='2020-01-01', enddate='2023-01-01')
# triggering automated statistical tests
test.run_tests()

# initialise class object for forecasting experiment for ITC stock data from yfinance
fin = Forecast(stock_code='ITC.NS', startdate='2020-01-01', enddate='2023-01-01')
# triggering the autoML for forecasting
result = fin.run()
```
The above code shows how to import the package and respective class methods to perform statistical tests and forecasting experiment

