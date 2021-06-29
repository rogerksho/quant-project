from alpha import generate_prediction_alpha, ALPHA
from beta import generate_prediction_beta, BETA

# impt var
CUTOFF = 2.0       # float (in %, 1-100)
PRINT_EVAL = True # bool

def main():
    # ticker and prediction lag
    ticker = "XLP"
    prediction_lag = 0

    print("==============================================================")
    beta_prediction = generate_prediction_beta(ticker, prediction_lag, PRINT_EVAL)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    alpha_prediction = generate_prediction_alpha(ticker, prediction_lag, PRINT_EVAL)
    alpha_beta_diff = alpha_prediction - beta_prediction

    print("==============================================================")
    print(f"probability of {BETA}%-drop in the next day:", beta_prediction)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"probability of {ALPHA}%-rise in the next day:", alpha_prediction)
    print("==============================================================")
    print("diff:", alpha_beta_diff)
    if alpha_beta_diff > CUTOFF:
        print("signal: BUY")
    elif alpha_beta_diff < -CUTOFF:
        print("signal: SELL")
    else:
        print("signal not strong enough")

if __name__ == "__main__":
    main()