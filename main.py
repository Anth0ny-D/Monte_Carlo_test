from monte_carlo_algorythm import MonteCarloAlgorythm
import argparse


def get_algo_params():
    pars = argparse.ArgumentParser()

    pars.add_argument('--i', type=str, default='true_labels_data.csv',
                      help='Path to csv with true labels')

    pars.add_argument('--n', type=int, default=100,
                      help='Number of iterations Monte Carlo tests')

    pars.add_argument('--o', type=str, default='results.png',
                      help='Path to save the plot')

    pars.add_argument('--p', nargs=3, type=float, default=[0.36, 0.12, 0.52],
                      help='Probability weights')

    params = pars.parse_args()
    return params


def main():
    a_params = get_algo_params()
    probs_algo = MonteCarloAlgorythm(data_path=a_params.i, probs=a_params.p, n=a_params.n, features=[0, 1, 2])
    probs_algo.plot_and_save_result(a_params.o)


if __name__ == "__main__":
    main()
