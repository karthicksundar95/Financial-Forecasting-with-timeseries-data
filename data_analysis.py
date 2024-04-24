""" Module to analysis financial data and perform automated statistical tests"""
# Necessary packages are imported
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf,acf,pacf

class StatisticalTests:
    """
    Class to run automated statistical tests
    """
    def __init__(self, data=None, colname='Adj Close',
                 stock_code=None, startdate=None, enddate=None):
        """
        Initialise all the class self variables
        :param data: python dataframe with time series data
        :param colname: name of column to work on for univariate time series
        :param ma_window_size: order size for moving average
        :param wma_weights: list of weights for weighted moving average
        :param stock_code: yahoo stock code for the stock to pull data from yfinance
        :param startdate: start data of data to pull from
        :param enddate: end data of the data range to pull from
        """
        self.stock_code = stock_code
        self.colname = colname
        if stock_code and startdate and enddate:
            self.start_date = startdate
            self.end_date = enddate
        elif data and self.colname:
            self.data = data[colname]
        else:
            raise Exception("Expected value for startdate and enddate if stock_code provided! [OR]\
                            Expectecd data and the column name to work on")
        self.result = {}
        self.writer = pd.ExcelWriter("fin_statistical_test_report.xlsx", engine='xlsxwriter')


    def fetch_data(self):
        """
        To capture the stock data from yahoo finance website
        :return: pandas dataframe with the data from the given time range
        """
        data = yf.download(self.stock_code, self.start_date, self.end_date)
        data.to_excel(self.writer, sheet_name='fetched_data')

        self.data = data[self.colname]
        workbook = self.writer.book
        worksheet = self.writer.sheets['fetched_data']
        format_ = workbook.add_format({'text_wrap': True})
        worksheet.set_column('A:K', None, format_)
        worksheet.autofit()

    def write_to_excel(self, sheet_name, colum_range,
                       img_file, img_position, test_summary_values=None):
        """
        To write the results to the excel to create the final report
        :param sheet_name: Name of the sheet
        :param colum_range: range of columns indices or names
        :param img_file: image file name for inserting the plot
        :param img_position: position of the image to put on the sheet
        :param test_summary_values: summary text to insert in the sheet
        :return: excel sheet with all data, results, text summary and plot
        """
        workbook = self.writer.book
        worksheet = self.writer.sheets[sheet_name]
        worksheet.hide_gridlines(2)
        format_ = workbook.add_format({'text_wrap': True})
        worksheet.set_column(colum_range, None, format_)
        worksheet.autofit()

        if test_summary_values:
            worksheet = self.insert_test_summary(worksheet, test_summary_values)
        worksheet.insert_image(img_position, img_file, {'x_scale': 0.25, 'y_scale': 0.25})

    def generate_plot(self, data, title, img_file, type_='line'):
        """
        Method to generate a plot for the tests
        :param data: time series plot in pandas dataframe format
        :param title: title for the plot
        :param img_file: name of the file to store the plot
        :param type_: type of plot to generate; default is line plot
        :return: plot image saved with the user provided filename
        """
        plt.figure()
        if type_ == 'line':
            data.plot(figsize=(50, 20), lw=4, legend=True)
            plt.title(title, fontdict={'fontsize': 50})
            plt.savefig(img_file)
        elif type_ == 'hist':
            plt.figure(figsize=(50, 20))
            plt.hist(data, bins=50)
            plt.title(title, fontdict={'fontsize': 50})
            plt.savefig(img_file)
        elif type_ == 'cf':
            f, ax = plt.subplots(nrows=2, ncols=1, figsize=(80, 40))
            plot_acf(data, lags=10, ax=ax[0])
            plot_pacf(data, lags=10, ax=ax[1], method='ols')
            plt.tight_layout()
            plt.savefig(img_file)

    def insert_test_summary(self, worksheet, values):
        """
        To insert summary for the statistical test
        :param worksheet: Name of the sheet to include the summary
        :param values: Values to be included in the summary
        :return: Text summary for the statistical test with its metric values
        """
        worksheet.insert_textbox(values[0], values[1], values[2], options={'line': {'none': True}})
        return worksheet

    def adfuller_test(self):
        """
        Method to run augmented dicker fuller test on the time series data
        :return: excel sheet with data, test results and plots
        """
        test_stats = adfuller(self.data)
        is_stationary = self.check_for_stationary(test_stats)
        df_diff = self.data
        diff_count = 0
        summary = "Augmented DICKER FULLER Test:\n\n" + str(test_stats)
        summary += f"\n\n Conclusion: \n Is the data series" \
                   f" stationary by default? : {is_stationary}\n"
        while is_stationary == False:
            diff_count += 1
            summary += f"To be differenced? : Yes\n\n # difference applied: {diff_count}\n\n"
            df_diff = self.make_data_stationary(df_diff)
            test_stats = adfuller(df_diff)
            summary += str(test_stats)
            is_stationary = self.check_for_stationary(test_stats)
            summary += f"\nConclusion: \n Is the data series stationary after differencing? : {is_stationary}\n"

        self.df_diff = df_diff
        self.data.to_excel(self.writer, sheet_name='stationary_test')

        self.generate_plot(self.data, f'{self.stock_code} stock actual time series', 'actual_ts.png')
        self.generate_plot(df_diff, f'{self.stock_code} stock diff time series', 'diff_ts.png')

        self.write_to_excel('stationary_test', 'A:K', 'actual_ts.png', 'H3', test_summary_values=[3, 3, summary])
        self.write_to_excel('stationary_test', 'A:K', 'diff_ts.png', 'H20')

    def make_data_stationary(self, data):
        """
        Method to convert non-stationary data to stationary nature by differencing
        :param data: data to be converted to stationary nature
        :return: converted data in dataframe format
        """
        df_log = np.log(data)
        df_diff = df_log.diff().dropna()
        return df_diff

    def check_for_stationary(self, test_stats):
        """
        To check if the data is complying with stationary properties
        :param test_stats: generated test statistics from the stationary test
        :return: Boolean indicating nature of data after evaluation
        """
        print("p-value:", test_stats[1])
        print("test_statistic", test_stats[0])
        if test_stats[1] > 0.05 and test_stats[0] > test_stats[4]['1%'] and test_stats[0] > test_stats[4]['5%'] and \
                test_stats[0] > test_stats[4]['10%']:
            print(
                "The test is statistically insignificant and hence we choose the NULL HYPOTHESIS (series is NOT STATIONARY)\n\n")
            return False
        else:
            print("The test is statistically significant and hence the series is STATIONARY)")
            return True

    def evaluate_normality(self, data, data_type):
        """
        Method to evaluate the normality nature of the data
        :param data: pandas dataframe
        :param data_type: indicator if the data is raw or differenced data
        :return: string output indicating normal distribution or not
        """
        summary = f'Kolmogorov\'s test for Normality:{data_type}\n\n'
        ktest_stats = stats.shapiro(data)
        summary += str(ktest_stats)
        print(ktest_stats)
        if ktest_stats[1] > 0.05:
            summary += "\n\n Conclusion: The data is NORMALLY distributed"
        else:
            summary += "\n\n Conclusion: The data is NOT NORMALLY distributed"

        return summary

    def kolmogorov_test(self):
        """
        Main method to execute and control the normality test
        :return: Normality test result is written to excel sheet with data, test statistic and plots
        """
        # actual adj close price
        self.data.to_excel(self.writer, sheet_name='Normality_test')

        actual_data_summary = self.evaluate_normality(self.data, data_type='actual')
        self.write_to_excel('Normality_test', 'A:K', 'actual_ts.png', 'H3',
                            test_summary_values=[3, 3, actual_data_summary])
        self.generate_plot(self.data, title='Distribution of actual data', img_file='Dist_of_actual_data.png',
                           type_='hist')
        self.write_to_excel('Normality_test', 'A:K', 'Dist_of_actual_data.png', 'H30')
        # print(self.df_diff)
        # try:
        # if self.df_diff:
        diff_data_summary = self.evaluate_normality(self.df_diff, data_type='diff_data')
        print(diff_data_summary)
        self.write_to_excel('Normality_test', 'A:K', 'diff_ts.png', 'H60',
                            test_summary_values=[60, 3, diff_data_summary])
        self.generate_plot(self.df_diff, title='Distribution of diff data', img_file='Dist_of_diff_data.png',
                           type_='hist')
        self.write_to_excel('Normality_test', 'A:K', 'Dist_of_diff_data.png', 'H90')

    def recommend_correlation_order(self, data):
        """
        To analyse the ACF and PACF plots and pick the best order for ARIMA model
        :param data: pandas dataframe with time series data
        :return: integer indicating order to be selected for AR and MA
        """
        acf_values = acf(data)
        pacf_values = pacf(data)
        pacf_counts = pd.Series(abs(pacf_values) > 0.05).value_counts().to_dict()
        acf_counts = pd.Series(abs(acf_values) > 0.05).value_counts().to_dict()
        print(abs(acf_values), abs(acf_values) > 0.05)
        if pacf_counts[True] > acf_counts[True]:
            return (f"Recommendation is MA{np.argmax(abs(acf_values[1:])) + 1}")
        else:
            return (f"Recommendation is AR{np.argmax(abs(pacf_values[1:])) + 1}")

    def correlaion_test(self):
        """
        To compute the correlation between the values of the time series data
        :return: ACF and PACF values and plot with recommended order values
        """
        self.data.to_excel(self.writer, sheet_name='Correlation_test')

        actual_summary = "Correlation test (ACF): actual data\n\n"
        self.generate_plot(self.data, title='ACF and PACF for actual data', img_file='actual_cf.png', type_='cf')
        actual_summary += str(self.recommend_correlation_order(self.data))
        self.write_to_excel('Correlation_test', 'A:K', 'actual_cf.png', 'H3',
                            test_summary_values=[3, 3, actual_summary])

        diff_summary = "Correlation test (ACF): diff data\n\n"
        self.generate_plot(self.df_diff, title='ACF and PACF for diff data', img_file='diff_cf.png', type_='cf')
        diff_summary += str(self.recommend_correlation_order(self.df_diff))

        self.write_to_excel('Correlation_test', 'A:K', 'diff_cf.png', 'H55',
                            test_summary_values=[55, 3, diff_summary])

    def run_tests(self):
        """
        Control method to execute the automated statistical tests
        :return: Final summary excel report with data, test statistics and plots
        """
        self.fetch_data()
        self.adfuller_test()
        self.kolmogorov_test()
        self.correlaion_test()
        self.writer.save()


