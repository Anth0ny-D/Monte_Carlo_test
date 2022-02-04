import random as r
from typing import List, Dict
import matplotlib.pyplot as plt
from os.path import exists


class MonteCarloAlgorythm:

    def __init__(self, data_path: str, probs: List[float], n: int, features: List[int]) -> None:
        self.true_labels_list = self.read_file(data_path=data_path)
        self.probs = probs
        self.n = n
        self.labels = features
        self.preds = self.make_predictions()
        self.metrics = self.get_final_metrics()
        assert sum(probs) == 1, 'sum of probability must be in 0..1'
        assert len(features) == 3, 'must be 3 features'

    def read_file(self, data_path: str) -> List[int]:
        # reading data. Inserted "try" because example csv file had first row as feature name

        assert exists(data_path) == True, 'Data path is wrong'
        with open(data_path, newline='') as file:
            data_list = []

            for row in file:
                data_list.append(int(row))

            assert len(data_list) > 0, 'Data is empty'

            return data_list

    def make_predictions(self) -> List[List[int]]:
        # function generates random choices tests
        predictions = []
        for i in range(0, self.n):
            monte_test = r.choices(self.labels, weights=self.probs, k=len(self.true_labels_list))
            predictions.append(monte_test)

        assert len(predictions) == self.n, 'comparing length of prediction and count of iterations failed'

        for pred in predictions:
            assert len(pred) == len(self.true_labels_list), 'comparing length of predicted labels to true labels data ' \
                                                            'failed '

        return predictions

    @staticmethod
    def accuracy(true_labels_list: List[int], predictions: List[int]) -> float:
        # accuracy = (TP+TN)/(TP+TN+FP+FN)

        assert len(true_labels_list) > 0, 'True labels list is empty'
        assert len(predictions) > 0, 'Predictions list is empty'
        assert len(true_labels_list) == len(predictions), 'Length of true labels isnt equal to length of predictions'

        true_positive_true_neg = (1 for i, z in zip(true_labels_list, predictions) if i == z)

        return sum(true_positive_true_neg) / len(true_labels_list)

    @staticmethod
    def precision(true_labels_list: List[int], predictions: List[int], class_number: int) -> float:
        # precision = TP/(TP+FP)

        assert len(true_labels_list) > 0, 'True labels list is empty'
        assert len(predictions) > 0, 'Predictions list is empty'
        assert len(true_labels_list) == len(predictions), 'Length of true labels isnt equal to length of predictions'

        true_positive = [(1 for true_item, pred_item in zip(true_labels_list, predictions) if
                          true_item == pred_item == class_number)]
        false_positive = [(1 for true_item, pred_item in zip(true_labels_list, predictions) if true_item != pred_item
                           and true_item == class_number)]

        return len(true_positive) / (len(true_positive) + len(false_positive))

    @staticmethod
    def recall(true_labels_list: List[int], predictions: List[int], class_number: int) -> float:
        # recall = TP/(TP+FN)

        assert len(true_labels_list) > 0, 'True labels list is empty'
        assert len(predictions) > 0, 'Predictions list is empty'
        assert len(true_labels_list) == len(predictions), 'Length of true labels isnt equal to length of predictions'

        true_positive = [(1 for true_item, pred_item in zip(true_labels_list, predictions) if
                          true_item == pred_item == class_number)]
        false_negative = [(1 for true_item, pred_item in zip(true_labels_list, predictions) if true_item == class_number
                           and pred_item != class_number)]

        return len(true_positive) / ((len(true_positive) + len(false_negative)))

    @staticmethod
    def get_cumulative_mean(input_list) -> List[float]:
        # basically function = np.cumsum[0:i]/i
        output_list = []
        element_to_append = input_list[0]

        for i in range(0, len(input_list)):
            if i == 0:
                output_list.append(element_to_append)
            else:
                element_to_append += input_list[i]
                output_list.append(element_to_append / (i + 1))
        assert len(input_list) > 0, 'Input list is empty'
        assert len(input_list) == len(output_list), 'Length of input list not equal to length of output list'

        return output_list

    def get_final_metrics(self) -> Dict[str, List[float]]:

        metrics = dict()
        metrics_list = ['Accuracy', 'Precision_for_0', 'Precision_for_1', 'Precision_for_2', 'Recall_for_0',
                        'Recall_for_1', 'Recall_for_2']
        accuracy_list = []
        precision_list_0 = []
        precision_list_1 = []
        precision_list_2 = []
        recall_list_0 = []
        recall_list_1 = []
        recall_list_2 = []

        for i in range(0, self.n):
            accuracy_list.append(self.accuracy(self.true_labels_list, self.preds[i]))
            precision_list_0.append(self.precision(self.true_labels_list, self.preds[i], 0))
            precision_list_1.append(self.precision(self.true_labels_list, self.preds[i], 1))
            precision_list_2.append(self.precision(self.true_labels_list, self.preds[i], 2))

            recall_list_0.append(self.recall(self.true_labels_list, self.preds[i], 0))
            recall_list_1.append(self.recall(self.true_labels_list, self.preds[i], 0))
            recall_list_2.append(self.recall(self.true_labels_list, self.preds[i], 0))
        general_results_list = [accuracy_list, precision_list_0, precision_list_1, precision_list_2,
                                recall_list_0, recall_list_1, recall_list_2]
        for i in range(0, len(general_results_list)):
            assert len(general_results_list[i]) == self.n, 'length of one of the results lists != count of iteractions'
            metrics[metrics_list[i]] = self.get_cumulative_mean(general_results_list[i])
        assert len(metrics) == 7, 'error: wrong count of metrics'

        return metrics

    def plot_and_save_result(self, output_path: str) -> None:
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:cyan', 'tab:olive', 'tab:orange', 'tab:purple']
        figure = plt.figure(figsize=(16, 10), dpi=80)
        for ind, i in enumerate(self.metrics):
            plt.plot(self.metrics[i], color=colors[ind], label=i)

        plt.xlim([0, (self.n - 1)])
        plt.xlabel(r'$Итерации$', fontsize=16)
        plt.ylabel(r'$Метрики$', fontsize=16)
        plt.legend(fontsize=14)

        plt.grid()
        figure.savefig(output_path)

        return plt.show()

# УБРАТЬ ФИЧИ ИЗ КЛАССА
# a = MonteCarloAlgorythm(data_path= 'true_labels_data_1.csv', probs=[0.5, 0.25, 0.25], n=5, features=[0,1,2])
