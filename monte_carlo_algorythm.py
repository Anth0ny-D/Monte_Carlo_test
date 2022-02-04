import random as r
from collections import defaultdict
from os.path import exists
from typing import List, Dict

import matplotlib.pyplot as plt


class MonteCarloAlgorythm:

    def __init__(self, data_path: str, probs: List[float], n: int) -> None:
        self.true_labels_list = self.read_file(data_path=data_path)
        self.probs = probs
        self.n = n
        self.preds = self.make_predictions()
        self.metrics = self.get_final_metrics()
        assert sum(probs) == 1, 'sum of probability must be in 0..1'

    def read_file(self, data_path: str) -> List[int]:
        """
        :param data_path: path to true labels
        :return: list of true labels
        """

        assert exists(data_path) == True, 'Data path is wrong'
        with open(data_path, newline='') as file:
            data_list = []

            for row in file:
                data_list.append(int(row))

            assert len(data_list) > 0, 'Data is empty'

            return data_list

    def make_predictions(self) -> List[List[int]]:
        """
        function generates random choices tests
        :return: list of lists with predictions
        """
        predictions = []
        for i in range(0, self.n):
            monte_test = r.choices([0, 1, 2], weights=self.probs, k=len(self.true_labels_list))
            predictions.append(monte_test)

        assert len(predictions) == self.n, 'comparing length of prediction and count of iterations failed'

        for pred in predictions:
            assert len(pred) == len(self.true_labels_list), 'comparing length of predicted labels to true labels data ' \
                                                            'failed '

        return predictions

    @staticmethod
    def accuracy(true_labels_list: List[int], predictions: List[int]) -> float:
        """
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        :param true_labels_list: list of true labels
        :param predictions: list of predictions (from make_predictions)
        :return: accuracy of predict (float)
        """

        assert len(true_labels_list) > 0, 'True labels list is empty'
        assert len(predictions) > 0, 'Predictions list is empty'
        assert len(true_labels_list) == len(predictions), 'Length of true labels isnt equal to length of predictions'

        true_positive_true_neg = (1 for i, z in zip(true_labels_list, predictions) if i == z)

        return sum(true_positive_true_neg) / len(true_labels_list)

    @staticmethod
    def precision(true_labels_list: List[int], predictions: List[int], class_number: int) -> float:
        """
        precision = TP/(TP+FP)
        :param true_labels_list: list of true labels
        :param predictions: list of predictions (from make_predictions)
        :param class_number: class of tested class
        :return: precision of predict (float)
        """

        assert len(true_labels_list) > 0, 'True labels list is empty'
        assert len(predictions) > 0, 'Predictions list is empty'
        assert len(true_labels_list) == len(predictions), 'Length of true labels isnt equal to length of predictions'

        true_positive = (true_item == pred_item == class_number for true_item, pred_item in
                         zip(true_labels_list, predictions))
        true_positive_and_false_positive = (pred_item == class_number for pred_item in predictions)

        return sum(true_positive) / sum(true_positive_and_false_positive)

    @staticmethod
    def recall(true_labels_list: List[int], predictions: List[int], class_number: int) -> float:
        """
        recall = TP/(TP+FN)
        :param true_labels_list: list of true labels
        :param predictions: list of predictions (from make_predictions)
        :param class_number: class of tested class
        :return: recall of predict (float)
        """

        assert len(true_labels_list) > 0, 'True labels list is empty'
        assert len(predictions) > 0, 'Predictions list is empty'
        assert len(true_labels_list) == len(predictions), 'Length of true labels isnt equal to length of predictions'

        true_positive = (true_item == pred_item == class_number for true_item, pred_item in
                         zip(true_labels_list, predictions))
        true_positive_and_false_negative = (pred_item == class_number for pred_item in true_labels_list)

        return sum(true_positive) / sum(true_positive_and_false_negative)

    @staticmethod
    def get_cumulative_mean(input_list) -> List[float]:
        """
        basically function = np.cumsum[0:i]/i
        :param input_list:
        :return:
        """

        cumulative_sum = 0
        output_list = []

        for i, item in enumerate(input_list, 1):
            cumulative_sum += item
            output_list.append(cumulative_sum / i)

        assert len(input_list) > 0, 'Input list is empty'
        assert len(input_list) == len(output_list), 'Length of input list not equal to length of output list'

        return output_list

    def get_final_metrics(self) -> Dict[str, List[float]]:
        """
        Creating one dict with results and transforms metrics to cumulative mean
        :return: dict[metric: cumulative means]
        """

        metrics = defaultdict(list)

        metrics_list = ['Accuracy', 'Precision_for_0', 'Precision_for_1', 'Precision_for_2', 'Recall_for_0',
                        'Recall_for_1', 'Recall_for_2']

        for i in range(0, self.n):
            metrics[metrics_list[0]].append(self.accuracy(self.true_labels_list, self.preds[i]))
            metrics[metrics_list[1]].append(self.precision(self.true_labels_list, self.preds[i], 0))
            metrics[metrics_list[2]].append(self.precision(self.true_labels_list, self.preds[i], 1))
            metrics[metrics_list[3]].append(self.precision(self.true_labels_list, self.preds[i], 2))

            metrics[metrics_list[4]].append(self.recall(self.true_labels_list, self.preds[i], 0))
            metrics[metrics_list[5]].append(self.recall(self.true_labels_list, self.preds[i], 1))
            metrics[metrics_list[6]].append(self.recall(self.true_labels_list, self.preds[i], 2))

        for i in metrics_list:
            metrics[i] = self.get_cumulative_mean(metrics[i])

        assert len(metrics) == 7, 'error: wrong count of metrics'

        return metrics

    def plot_and_save_result(self, output_path: str) -> None:
        """
        Plotting all results in one file
        :param output_path:
        :return: plot on screen and saving to path
        """
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:cyan', 'tab:olive', 'tab:orange', 'tab:purple']
        fig, ax = plt.subplots(len(self.metrics), 1, figsize=(16, 25), dpi=80)

        for ind, i in enumerate(self.metrics):
            ax[ind].plot(range(self.n), self.metrics[i], color=colors[ind])
            ax[ind].set_title(i)
            ax[ind].grid()
            fig.savefig(output_path)

        return plt.show()
