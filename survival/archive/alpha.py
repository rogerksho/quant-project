import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta import momentum

from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter

# impt vars
ALPHA = 1
TRAINING_PERIOD = "8y" # note format


# visualisation stuff
# aft.plot_partial_effects_on_outcome(covariates='acc_ROC', values=np.arange(-5,5), cmap='coolwarm')
# aft.plot_partial_effects_on_outcome(covariates='avg_ROC', values=np.arange(-1,1,.3), cmap='coolwarm')
# plt.show()


def generate_prediction_alpha(ticker, prediction_lag, print_eval):
    TRAINING_END_DATE = '2021-03-01'

    # download data
    spy = yf.Ticker(ticker)
    hist_spy = spy.history(period=TRAINING_PERIOD)
    hist_spy = hist_spy.loc[: TRAINING_END_DATE]

    VIX = yf.Ticker("^VIX")
    hist_VIX = VIX.history(period=TRAINING_PERIOD)["Close"]
    VIX_cropped = hist_VIX.loc[hist_spy.index[0] : hist_spy.index[-1]]

    # calculate pct change
    hist_spy["pct_change_close"] = hist_spy["Close"].pct_change() * 100
    hist_spy = hist_spy.iloc[1:]

    alpha_rise = pd.DataFrame(hist_spy[hist_spy["pct_change_close"] >= ALPHA]["pct_change_close"])
    alpha_duration = pd.Series(alpha_rise.index) - pd.Series(alpha_rise.index).shift(1)
    # date attached is the date of event happening, duration is time since last event
    alpha_rise = alpha_rise.assign(abs_duration=alpha_duration.values)
    alpha_rise = alpha_rise.iloc[1:]

    # insert column of dates of last event trigger
    alpha_rise["prev_date"] = alpha_rise.index - alpha_rise["abs_duration"]

    # calculate acc_ROC, avg_ROC
    active_days_series = []
    acc_ROC_series = []
    avg_ROC_series = []
    exp_avg_VIX_series = []
    rsi_series = []
    psy_series = []

    for idx, row in alpha_rise.iterrows():
        # define date ranges for calculation, minus 1 day from end date to avoid look-ahead bias
        start_date = row["prev_date"]
        end_date = idx - np.timedelta64(1, 'D')

        acc_ROC = hist_spy.loc[start_date : end_date]["pct_change_close"].sum()
        acc_ROC_series.append(acc_ROC)

        active_days = hist_spy.loc[start_date : end_date]["pct_change_close"].shape[0]
        avg_ROC = hist_spy.loc[start_date : end_date]["pct_change_close"].ewm(span=active_days).mean()[-1]

        avg_VIX = VIX_cropped.loc[row["prev_date"] : idx - np.timedelta64(1, 'D')].ewm(span=active_days).mean()
        exp_avg_VIX_series.append(avg_VIX[-1]) # most recent ema

        psy_line = sum(hist_spy.loc[start_date : end_date]["pct_change_close"] > 0)/active_days
        psy_series.append(psy_line)

        rsi_series.append(momentum.rsi(hist_spy.loc[start_date : end_date]["Close"], window=active_days)[-1])
        active_days_series.append(active_days)
        avg_ROC_series.append(avg_ROC)

    # mean time between events
    # print("mean: ", np.mean(active_days_series))

    # create new dataframe for fitting
    rise_df = pd.DataFrame({'duration': active_days_series, 'acc_ROC': acc_ROC_series, 'avg_ROC': avg_ROC_series, 'exp_avg_VIX': exp_avg_VIX_series, 'rsi': rsi_series, 'psy': psy_series})

    # fit with weibull aft
    aft = CoxPHFitter()
    aft.fit(rise_df, duration_col='duration')

    # print fitting summary
    if print_eval: 
        aft.print_summary()

    '''
    how to interpret:
    - exp(coef) tells you the hazard ratio (HR); HR = 1 no effect, HR > 1 increase hazard, HR < 1 reduce hazard
    - AIC, the lower the better (should only compare between models trained on the same dataset)
    '''


    # generate prediction data
    latest_date = alpha_rise.index[-1]
    today_date = np.datetime64('today') - np.timedelta64(prediction_lag, 'D')

    acc_ROC_test = hist_spy.loc[latest_date : today_date]["pct_change_close"].sum()

    active_days_test = hist_spy.loc[latest_date : today_date]["pct_change_close"].shape[0]
    avg_ROC_test = hist_spy.loc[latest_date : today_date]["pct_change_close"].ewm(span=active_days_test).mean()[-1]

    exp_avg_VIX_test = VIX_cropped.loc[latest_date : today_date].ewm(span=active_days_test).mean()[-1] # most recent ema

    rsi_test = momentum.rsi(hist_spy.loc[latest_date : today_date]["Close"], window=active_days_test)[-1]

    psy_test = sum(hist_spy.loc[latest_date : today_date]["pct_change_close"] > 0)/active_days_test

    # testing data logging
    print(f"days since last {ALPHA}%-rise:", active_days_test)
    # print("acc_ROC:", acc_ROC_test)
    # print("avg_ROC:", avg_ROC_test)

    ####### prediction ########

    test_df = pd.DataFrame({'acc_ROC': [acc_ROC_test], 'avg_ROC': [avg_ROC_test], 'exp_avg_VIX': [exp_avg_VIX_test], 'rsi': [rsi_test], 'psy': [psy_test]})
    prediction = aft.predict_survival_function(test_df, conditional_after=[active_days_test])

    # logging
    print("latest data:", VIX_cropped.loc[latest_date : today_date].index[-1])

    # return 1 - S(T_c - T_a), which is the probability of the event happening the next day
    return(100*(1 - prediction[0].iloc[0]))


if __name__ == "__main__":
    print(f"probability of {ALPHA}%-rise in the next day:", generate_prediction_alpha("SPY", 0, False))

