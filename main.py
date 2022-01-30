from Probs_algo import *
import argparse


def get_algo_params():
    pars = argparse.ArgumentParser()

    pars.add_argument('--i', type=str, default='true_labels_data.csv',
                      help='Path to csv with true labels. If there is no file - creates with default parameters:'
                           'DEFAULT_LABELS = [0, 1, 2]'
                           'DEFAULT_LENGTH = 10000'
                           'DEFAULT_WEIGHTS = [0.36, 0.12, 0.52]'
                           'N = 100')

    pars.add_argument('--n', type=int, default=N,
                        help='Number of iterations Monte Carlo tests')

    pars.add_argument('--o', type=str, default='results.png',
                        help='Path to save the plot')

    pars.add_argument('--p', nargs=3, type=float, default=DEFAULT_WEIGHTS,
                        help='Probability weights')

    params = pars.parse_args()
    return params


def main():
    a_params = get_algo_params()
    try:

        probs_algo = Probs_algo(data_path=a_params.i, probs=a_params.p, n=a_params.n, labels=DEFAULT_LABELS)

    except: create_true_labels()

    probs_algo = Probs_algo(data_path=a_params.i, probs=a_params.p, n=a_params.n, labels=DEFAULT_LABELS)
    probs_algo.plot_and_save_result(a_params.o)


if __name__ == "__main__":
    main()