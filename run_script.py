""" Test script to execute and mimic user action of using the FinAutoML package"""

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

