import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta import momentum, volatility

from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
        
# class to store cph model and dataframes
class Model:
    def __init__(self, duration_df, cph_model):
        self.duration_df = duration_df
        self.cph_model = cph_model

    def get_last_event_date_training(self):
        return(self.duration_df.index[-1])

    def get_prediction(self):
        # generate prediction
        current_duration = self.test_df["current_duration"]
        self.test_df = self.test_df.drop(columns=["current_duration"], axis=1)

        prediction = self.cph_model.predict_survival_function(self.test_df, conditional_after=[current_duration])

        return(100*(1 - prediction[0].iloc[0]))

# specify either ALPHA or BETA (in %, 1-100) but not both to decide to calculate rise/drop
def generate_duration_df(prices_df, ALPHA=None, BETA=None): 
    # calculate pct change and crop the NaN row
    prices_df = prices_df.iloc[1:]

    if ALPHA is not None and BETA is not None:
        raise Exception("Please only specify either ALPHA or BETA but not both.")
    elif ALPHA is not None:
        # BETA is None
        assert(BETA is None)
        # calculate duration df
        duration_df = pd.DataFrame(prices_df[prices_df["pct_change_close"] >= ALPHA]["pct_change_close"])
    else:
        # ALPHA is None
        assert(ALPHA is None)
        # calculate duration df
        duration_df = pd.DataFrame(prices_df[prices_df["pct_change_close"] <= -BETA]["pct_change_close"])

    # insert column with previous dates of event triggers
    duration_df = duration_df.assign(prev_date=pd.Series(duration_df.index).shift(1).values)

    # crop NaT row
    duration_df = duration_df.iloc[1:]

    return duration_df


def generate_training_df(duration_df, prices_df): 
    # calculate pct change and crop the NaN row
    prices_df = prices_df.iloc[1:]

    # covariate series
    active_days_series = []

    acc_ROC_series = []
    avg_ROC_series = []

    rsi_series = []
    psy_series = []

    atr_series = []

    for idx, row in duration_df.iterrows():
        # define date ranges for calculation, minus 1 day from end date to avoid look-ahead bias
        start_date = row["prev_date"]
        end_date = idx - np.timedelta64(1, 'D')

        prices_df_cropped = prices_df.loc[start_date : end_date]

        # calculate indicators
        acc_ROC = prices_df_cropped["pct_change_close"].sum()

        active_days = prices_df_cropped["pct_change_close"].shape[0]
        avg_ROC = prices_df_cropped["pct_change_close"].ewm(span=active_days).mean()[-1]

        psy_line = (sum(prices_df_cropped["pct_change_close"] > 0) / active_days) * 100

        atr_obj = volatility.AverageTrueRange(high=prices_df_cropped["High"], low=prices_df_cropped["Low"], close=prices_df_cropped["Close"], window=active_days)
        atr_val = atr_obj.average_true_range()[-1]

        # append to series
        active_days_series.append(active_days)
        
        acc_ROC_series.append(acc_ROC)
        avg_ROC_series.append(avg_ROC)

        rsi_series.append(momentum.rsi(prices_df_cropped["Close"], window=active_days)[-1])
        psy_series.append(psy_line)

        atr_series.append(atr_val)

    training_df = pd.DataFrame({'duration': active_days_series, 'acc_ROC': acc_ROC_series, 'avg_ROC': avg_ROC_series, 'rsi': rsi_series, 'psy': psy_series, 'atr': atr_series})

    return(training_df)

def train_model(training_df, print_eval: bool): # print_eval = True will print out training summary
    # fit with cox proportional hazard model
    cph = CoxPHFitter()
    cph.fit(training_df, duration_col='duration')

    # print fitting summary
    if print_eval: 
        cph.print_summary()

    '''
    how to interpret:
    - exp(coef) tells you the hazard ratio (HR); HR = 0 no effect, HR > 1 increase hazard, HR < 1 reduce hazard
    - AIC, the lower the better (should only compare between models trained on the same dataset)
    '''

    return(cph)

def generate_test_df(prices_df, last_hit_date, current_date):
    # crop df from last hit date to current date
    prices_df = prices_df.loc[last_hit_date : current_date]

    # active trading days in current sample
    active_days_test = prices_df.shape[0]

    # calculate data
    acc_ROC_test = prices_df["pct_change_close"].sum()
    avg_ROC_test = prices_df["pct_change_close"].ewm(span=active_days_test).mean()[-1]
    rsi_test = momentum.rsi(prices_df["Close"], window=active_days_test)[-1]
    psy_line_test = (sum(prices_df["pct_change_close"] > 0) / active_days_test) * 100

    atr_obj_test = atr_obj = volatility.AverageTrueRange(high=prices_df["High"], low=prices_df["Low"], close=prices_df["Close"], window=active_days_test)
    atr_val_test = atr_obj_test.average_true_range()[-1]

    # organize in df
    test_df = pd.DataFrame({'current_duration': [active_days_test], 'acc_ROC': [acc_ROC_test], 'avg_ROC': [avg_ROC_test], 'rsi': [rsi_test], 'psy': [psy_line_test], 'atr': [atr_val_test]})

    return(test_df)


if __name__ == "__main__":
    generate_duration_df(None, BETA=1, ALPHA=1)
