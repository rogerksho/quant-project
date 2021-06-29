import pandas as pd 
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from collections import defaultdict

# impt var
TOP_N = 2
STDEV_CUTOFF = 1.0

# portfolio class
class Portfolio:
    def __init__(self, index_df_dict, spy_hist, vix_hist, starting_capital):
        self.holdings = dict()
        self.capital = starting_capital
        self.index_df_dict = index_df_dict
        self.spy_hist = spy_hist
        self.vix_hist = vix_hist

        self.performance_df = pd.DataFrame(columns=['spy_benchmark', 'performance'])
        self.max_drawdown = ['NaT', 0]
        self.winners_macd_df = pd.DataFrame()

        self.std_df = pd.DataFrame()

        self.holdings_df = pd.DataFrame()

        self.compute_macd_std()

    def compute_macd_std(self):
        macd_df = pd.DataFrame()

        for index_df in self.index_df_dict.items():
            macd_df = macd_df.join(index_df[1]["MACD_SMA"], rsuffix=f'_{index_df[0]}', how='outer')

        # calculate stdev
        self.std_df["stdev"] = macd_df.std(axis=1) ** 2

        # further process
        self.std_df["stdev_sma_5"] = self.std_df["stdev"].rolling(5).mean()
        self.std_df["stdev_sma_63"] = self.std_df["stdev"].rolling(63).mean()
        self.std_df["stdev_sma_ratio"] = self.std_df["stdev_sma_5"] / self.std_df["stdev_sma_63"]

    def prev_workday(self, date):
        return(self.spy_hist.iloc[self.spy_hist.index.get_loc(date) - 1].name)

    def prev_month(self, date):
        date -= np.timedelta64(45, 'D')
        date += pd.tseries.offsets.MonthEnd(1)

        return(date)

    def init_benchmark_spy(self, date):
        self.spy_num_shares = self.capital // self.spy_hist.loc[date]["Close"]

    def rebalance(self, date):
        # logging
        print("=========================")
        print("date:", date)

        # preventing look-ahead bias
        prev_date = self.prev_workday(date)

        # risk adjusted n
        risk_adjusted_n = 0

        # if still in market, find current gainers
        #current_gainers = self.find_top_gainers_stoch_RSI(prev_date)
        #current_gainers = self.find_top_gainers_rolling_avg(prev_date)
        #current_gainers = self.find_top_gainers_RSI(prev_date)
        #current_gainers = self.find_top_gainers_MACD(prev_date)
        #current_gainers = self.find_top_gainers_linear(prev_date)
        #current_gainers = self.find_top_gainers_past_n_months(prev_date, 5)
        current_gainers = self.find_top_gainers_variable_lookback(prev_date)

        # exit market if SPY dips below 10m SMA
        #if self.spy_hist.loc[prev_date]["Close"] < self.spy_hist.loc[prev_date]["10m_SMA"] or self.std_df.loc[prev_date]["stdev_sma_ratio"] > STDEV_CUTOFF:
        if self.spy_hist.loc[prev_date]["Close"] < self.spy_hist.loc[prev_date]["10m_EMA"]:
            self.exit_all(date)
            # record holdings if exiting
            self.record_holdings(date)
            return

        # enter market if empty
        if not len(self.holdings):
            print("entering market...")
            for etf in current_gainers:
                self.buy(etf, date)
            # record holdings if entering market
            self.record_holdings(date)
            return

        # rebalancing here if you are already holding stuff
        to_sell = list(set(self.holdings.keys()) - set(current_gainers))
        to_buy = list(set(current_gainers) - set(self.holdings.keys()))

        # make sure sell.count == buy.count, then sell -> buy
        assert(len(to_sell) == len(to_buy))

        for sell_etf in to_sell:
            self.sell(sell_etf, date)

        for buy_etf in to_buy:
            self.buy(buy_etf, date)

        if not len(to_buy):
            print("holding...")

        # record holdings after rebalance
        self.record_holdings(date)

        print("balance:", self.capital)
        self.calculate_drawdown(date)

        return

    def exit_all(self, date):
        # if weekend, sell next market open
        if date not in self.spy_hist.index:
            date = self.spy_hist.loc[date : date + np.timedelta64(7, 'D')].index[0]

        # if holdings empty
        if not len(self.holdings):
            print("nothing to do...")

        else:
            print("exiting market...")
            for etf in list(self.holdings):
                self.sell(etf, date)


    def buy(self, ticker, date):
        # get price
        price = self.index_df_dict[ticker].loc[date]["Close"]

        # buy
        partitioned_capital = self.capital // (TOP_N - len(self.holdings))
        share_count = partitioned_capital // price
        actual_total_price = share_count * price

        print("buying", share_count, "shares of", ticker, "for", price, "USD, total:", actual_total_price)

        # deduct money
        self.holdings[ticker] = share_count
        self.capital -= actual_total_price

        # make sure balance is nonnegative
        assert(self.capital > 0)

    def sell(self, ticker, date):
        # get price
        price = self.index_df_dict[ticker].loc[date]["Close"]

        # sell
        total_sale = self.holdings[ticker] * price

        print("selling", self.holdings[ticker], "shares of", ticker, "for", price, "USD, total:", total_sale)

        # add money
        del self.holdings[ticker]
        self.capital += total_sale

        # make sure balance is nonnegative
        assert(self.capital > 0)

    def record_asset_value(self, date):
        actual_balance = self.capital

        for etf_tuple in self.holdings.items():
            ticker = etf_tuple[0]
            share_count = etf_tuple[1]

            price = self.index_df_dict[ticker].loc[date]["Close"]
            actual_balance += price * share_count

        # total asset value before rebalancing
        new_row = pd.DataFrame(data={'spy_benchmark': self.spy_num_shares * self.spy_hist.loc[date]["Close"], 'performance': actual_balance}, index=[date])
        self.performance_df = self.performance_df.append(new_row, verify_integrity=True)

        return(actual_balance)

    def record_holdings(self, date):
        index_list = list(self.index_df_dict)
        current_holdings_dict = dict()

        for index in index_list:
            current_holdings_dict[index] = int(index in self.holdings)

        new_row = pd.DataFrame(data=current_holdings_dict, index=[date])
        self.holdings_df = self.holdings_df.append(new_row, verify_integrity=True)

    # TODO: given a date, find the top N performing indices in the last month
    def get_past_winners(self, date):
        lookback_date = self.prev_month(date)

        # account for weekends
        lookback_date_adj = self.spy_hist.loc[lookback_date - np.timedelta64(7, 'D') : lookback_date].index[-1]

        # start search
        past_gainers_dict = {}

        for index_df_tuple in self.index_df_dict.items():
            pct_change = (index_df_tuple[1].loc[date]["Close"] - index_df_tuple[1].loc[lookback_date_adj]["Close"]) / index_df_tuple[1].loc[lookback_date_adj]["Close"]
            past_gainers_dict[index_df_tuple[0]] = pct_change

        past_gainers_dict = dict(sorted(past_gainers_dict.items(), key=lambda item: item[1], reverse=True))

        return(list(past_gainers_dict)[:TOP_N])

    def get_asset_value(self, date):
        return(self.performance_df.loc[date]["performance"])

    def plot_performance(self):
        # subplots
        fig, ax = plt.subplots(5, sharex=True)
        
        # subplot 1
        self.performance_df.plot(ax=ax[0])
        ax[0].text('2007-12-01', 170000, "2007-08 Recession", color='red')
        ax[0].legend(loc=3)

        # subplot 2
        self.yearly_performance()[["benchmark", "performance"]].plot(ax=ax[1])
        ax[1].axhline(color='r', linewidth=0.5)

        # subplot 3
        self.monthly_performance()[["benchmark_monthly", "performance_monthly"]].plot(ax=ax[2])
        ax[2].axhline(color='r', linewidth=0.5)
        ax[2].legend(loc=3)

        # subplot 4
        self.vix_hist["sma"].plot(ax=ax[3])
        #ax[3].axhline(y=STDEV_CUTOFF, color='r', linewidth=0.5)

        # subplot 5
        self.std_df.drop(columns=["stdev_sma_ratio"]).plot(ax=ax[4])

        print(self.holdings_df.sum())

        plt.show()

    def calculate_drawdown(self, date):
        recent_peak = self.performance_df["performance"].loc[: date].max()

        # most recent performance
        recent_performance = self.performance_df["performance"][-1]

        # if most recent performance is below recent peak, calculate drawdown
        if recent_performance < recent_peak:
            drawdown = (recent_performance - recent_peak)/recent_peak

            if drawdown < self.max_drawdown[1]:
                self.max_drawdown[1] = drawdown
                self.max_drawdown[0] = date.strftime("%Y-%m-%d")

            return(drawdown)
        return(0)

    def get_ensemble_results(self, date):
        ensemble_dict = dict()
        prev_date = self.prev_date(date)

        ensemble_dict["MACD_gainers"] = find_top_gainers_MACD(prev_date)
        ensemble_dict["RSI_gainers"] = find_top_gainers_RSI(prev_date)
        ensemble_dict["rolling_avg_gainers"] = find_top_gainers_rolling_avg(prev_date)
        ensemble_dict["stoch_RSI_gainers"] = find_top_gainers_stoch_RSI(prev_date)
        ensemble_dict["linear_gainers"] = find_top_gainers_linear(prev_date)

        return(ensemble_dict)

    def find_top_gainers_MACD(self, date):
        top_gainers_dict = {}

        for index_df in self.index_df_dict.items():
            top_gainers_dict[index_df[0]] = index_df[1].loc[date]["MACD_SMA"]

        # find top N
        top_gainers_dict = dict(sorted(top_gainers_dict.items(), key=lambda item: item[1], reverse=True))

        # stdev of MACD
        gainers_stdev = pd.Series(top_gainers_dict.values()).std() ** 2

        return(list(top_gainers_dict)[:TOP_N])

    def find_top_gainers_stoch_RSI(self, date):
        top_gainers_dict = {}

        for index_df in self.index_df_dict.items():
            top_gainers_dict[index_df[0]] = index_df[1].loc[date]["stoch_RSI"]

        # find top N
        top_gainers_dict = dict(sorted(top_gainers_dict.items(), key=lambda item: item[1], reverse=True))

        return(list(top_gainers_dict)[:TOP_N])

    def find_top_gainers_variable_lookback(self, date):
        date_list = [1, 3, 6, 9, 12]
        result_dict = defaultdict(int)

        for idx, val in enumerate(date_list):
            for etf in self.find_top_gainers_past_n_months(date, val):
                result_dict[etf] += 1

        # find top N
        sorted_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))

        return(list(sorted_dict)[:TOP_N])

    def find_top_gainers_past_n_months(self, date, n):
        top_gainers_dict = {}
        lookback_date = self.spy_hist.index[self.spy_hist.index.get_loc(date) - int(21*n)]

        for index_df in self.index_df_dict.items():
            top_gainers_dict[index_df[0]] = (index_df[1]["Close"].loc[date] - index_df[1]["Close"].loc[lookback_date]) / index_df[1]["Close"].loc[lookback_date]

        # find top N
        top_gainers_dict = dict(sorted(top_gainers_dict.items(), key=lambda item: item[1], reverse=True))

        return(list(top_gainers_dict)[:TOP_N])


    def find_top_gainers_linear(self, date):
        lin_reg_lookback = 21
        top_gainers_dict = {}

        for index_df in self.index_df_dict.items():
            recent_prices = index_df[1].iloc[index_df[1].index.get_loc(date) - lin_reg_lookback : index_df[1].index.get_loc(date)]["Close"]
            reg = LinearRegression().fit(recent_prices.values.reshape(-1, 1), np.arange(recent_prices.shape[0]))
            top_gainers_dict[index_df[0]] = reg.coef_[0]

        
        # find top N
        top_gainers_dict = dict(sorted(top_gainers_dict.items(), key=lambda item: item[1], reverse=True))
        return(list(top_gainers_dict)[:TOP_N])

    def find_top_gainers_RSI(self, date):
        top_gainers_dict = {}

        for index_df in self.index_df_dict.items():
            top_gainers_dict[index_df[0]] = index_df[1].loc[date]["RSI_SMA"]

        # find top N
        top_gainers_dict = dict(sorted(top_gainers_dict.items(), key=lambda item: item[1], reverse=True))

        # return top gainers
        return(list(top_gainers_dict)[:TOP_N])

    def find_top_gainers_rolling_avg(self, date):
        top_gainers_dict = {}

        for index_df in self.index_df_dict.items():
            top_gainers_dict[index_df[0]] = index_df[1].loc[date]["rolling_avg_diff"]

        # find top N
        top_gainers_dict = dict(sorted(top_gainers_dict.items(), key=lambda item: item[1], reverse=True))

        # return top gainers
        return(list(top_gainers_dict)[:TOP_N])

    def sharpe_ratio(self):
        # monthly performance
        monthly_performance_df = self.monthly_performance()

        # load risk free rate data
        rfr = pd.read_csv("../data/1m_rfr.csv", index_col="DATE", dtype={'DGS1MO': np.float64}, na_values='.', parse_dates=True)

        # join
        monthly_performance_df = monthly_performance_df.join(rfr, how='left')

        # calculate diff
        monthly_performance_df["diff"] = 12*100*monthly_performance_df["performance_monthly"] - monthly_performance_df["DGS1MO"]
        monthly_performance_df["benchmark_diff"] = 12*100*monthly_performance_df["benchmark_monthly"] - monthly_performance_df["DGS1MO"]

        return_stdev = monthly_performance_df["diff"].std()
        return_mean = monthly_performance_df["diff"].mean()

        benchmark_stdev = monthly_performance_df["benchmark_diff"].std()
        benchmark_mean = monthly_performance_df["benchmark_diff"].mean()

        print("std_bench", benchmark_stdev)
        print("std_strat", return_stdev)

        print("benchmark sharpe:", benchmark_mean / benchmark_stdev)

        return(return_mean / return_stdev)

    def monthly_performance(self):
        # monthly performance df
        monthly_performance_df = pd.DataFrame()

        # start on month end
        start_date = self.spy_hist.index[0]
        start_date += pd.offsets.MonthEnd(1)

        while start_date < np.datetime64('today'):
            # adjust for weekends
            if start_date not in self.spy_hist.index:
                if self.spy_hist.loc[start_date :].shape[0] == 0:
                    break
                start_date = self.spy_hist.loc[start_date :].index[0]
                continue
    
            try:
                if not start_date.month == self.spy_hist.iloc[self.spy_hist.index.get_loc(start_date) + 1].name.month:
                    # append to big df
                    new_row = pd.DataFrame({'monthly_performance': self.performance_df["performance"].loc[start_date], 'spy_benchmark': self.performance_df["spy_benchmark"].loc[start_date]}, index=[start_date])
                    monthly_performance_df = monthly_performance_df.append(new_row)
            # pass on IndexError thrown on last month
            except IndexError:
                pass

            start_date += np.timedelta64(1, 'D')

        # calculate pct change
        monthly_performance_df["performance_monthly"] = monthly_performance_df["monthly_performance"].pct_change() * 100
        monthly_performance_df["benchmark_monthly"] = monthly_performance_df["spy_benchmark"].pct_change() * 100

        return(monthly_performance_df)

    def yearly_performance(self):
        # monthly performance df
        yearly_performance_df = pd.DataFrame()

        # start on month end
        start_date = self.spy_hist.index[0]
        start_date += pd.offsets.YearEnd(1)

        while start_date < np.datetime64('today'):
            if start_date not in self.spy_hist.index:
                start_date = self.spy_hist.loc[start_date : start_date + np.timedelta64(7, 'D')].index[0]

            # append to big df
            new_row = pd.DataFrame({'yearly_performance': self.performance_df["performance"].loc[start_date], 'spy_benchmark': self.performance_df["spy_benchmark"].loc[start_date]}, index=[start_date])
            yearly_performance_df = yearly_performance_df.append(new_row)

            start_date += np.timedelta64(7, 'D')
            start_date += pd.offsets.YearEnd(1)

        yearly_performance_df["performance"] = yearly_performance_df["yearly_performance"].pct_change() * 100
        yearly_performance_df["benchmark"] = yearly_performance_df["spy_benchmark"].pct_change() * 100

        return(yearly_performance_df)

