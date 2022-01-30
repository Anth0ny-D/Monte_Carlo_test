import random as r
import pandas as pd

import typing as tp
import matplotlib.pyplot as plt


DEFAULT_LABELS = [0, 1, 2]
DEFAULT_LENGTH = 10000
DEFAULT_WEIGHTS = [0.36, 0.12, 0.52]
N = 100
List = tp.List
Dict = tp.Dict
Tuple = tp.Tuple


def create_true_labels():
    true_labels_data = r.choices(DEFAULT_LABELS, weights=DEFAULT_WEIGHTS, k=DEFAULT_LENGTH)

    df = pd.DataFrame(true_labels_data, columns=['true_labels'])
    df.to_csv('true_labels_data.csv', index=False)

    return len(df)



class Probs_algo:

    def __init__(self, data_path: str, probs: List[float], n: int, labels: List[int]) -> None:
        self.true_labels_list = self.read_file(data_path=data_path)
        self.probs = probs
        self.n = n
        self.labels = labels
        self.preds = self.make_predictions()
        self.metrics = self.get_final_metrics()

    def read_file(self, data_path: str) -> List[int]:

        with open(data_path) as file:
            data = pd.read_csv(file)
            self.data_list = list(data[data.columns[0]])

            return self.data_list

    def make_predictions(self) -> List[List[int]]:

        predictions = []
        for i in range(0, self.n):
            monte_test = r.choices(self.labels, weights=self.probs, k=len(self.true_labels_list))
            predictions.append(monte_test)

        assert len(predictions) == self.n

        for pred in predictions:
            assert len(pred) == len(self.true_labels_list)
        return predictions

    @staticmethod
    # accuracy = (TP+TN)/(TP+TN+FP+FN) при дисбалансе классов может показывать чушь

    def accuracy(true_labels_list: List[int], predictions: List[int]) -> Dict[int, float]:
        accuracy_dict = {}
        for high_iteration_index, prediction_list in enumerate(predictions):
            x = 0
            prediction_list: tp.Iterable
            for low_iteration_index, pred_item in enumerate(prediction_list):
                if true_labels_list[low_iteration_index] == pred_item:
                    x += 1
            accuracy_dict[high_iteration_index] = x / len(true_labels_list)
        # print(accuracy_dict)
        return accuracy_dict

    @staticmethod
    def precision_and_recall(true_labels_list: List[int], predictions: List[int], class_number: int) -> Tuple[
        Dict[int, float], Dict[int, float]]:
        # precision = TP/(TP+FP) определяет точность определения конкретного класса,
        # относительно всех объектов которые система отнесла к этому классу.
        precision_dict = {}

        # Полнота (recall) демонстрирует способность алгоритма обнаруживать данный класс вообще. recall = TP/(TP+FN)
        recall_dict = {}

        for high_iteration_index, prediction_list in enumerate(predictions):
            TP = 0
            FP = 0
            FN = 0
            prediction_list: tp.Iterable

            for low_iteration_index, pred_item in enumerate(prediction_list):
                if true_labels_list[low_iteration_index] == pred_item == class_number:
                    TP += 1
                if pred_item == class_number != true_labels_list[low_iteration_index]:
                    FP += 1
                if pred_item != class_number == true_labels_list[low_iteration_index]:
                    FN += 1

            precision_dict[high_iteration_index] = TP / (TP + FP)
            recall_dict[high_iteration_index] = TP / (TP + FN)
        # print(precision_dict, recall_dict)

        return precision_dict, recall_dict

    def get_cumulative_mean(self, list_to_change) -> List[float]:
        new_list = []
        for i in range(0, len(list_to_change)):
            if i == 0:
                new_list.append(round((list_to_change[i]), 5))
            else:
                new_list.append(round(sum(list_to_change[:i]) / i, 5))
        return new_list

    def get_final_metrics(self) -> Dict[str, List[float]]:
        metrics = dict()
        accuracy_list = []
        full_data = []

        Accuracy = self.accuracy(true_labels_list=self.true_labels_list, predictions=self.preds)

        for i in Accuracy:
            accuracy_list.append(Accuracy[i])

        for k in self.labels:
            Precision = list(self.precision_and_recall(true_labels_list=self.true_labels_list,
                                                       predictions=self.preds,
                                                       class_number=k)[0].values())
            Recall = list(self.precision_and_recall(true_labels_list=self.true_labels_list,
                                                    predictions=self.preds,
                                                    class_number=k)[1].values())
            full_data.append(Precision)
            full_data.append(Recall)
        cumulative_acc = self.get_cumulative_mean(accuracy_list)
        metrics_list = ['Precision_for_0', 'Recall_for_0', 'Precision_for_1', 'Recall_for_1',
                        'Precision_for_2', 'Recall_for_2']
        for ind, i in enumerate(full_data):
            metrics[metrics_list[ind]] = self.get_cumulative_mean(i)
        metrics['Accuracy'] = cumulative_acc
        # print(metrics)

        assert len(metrics) == 7

        for metric in metrics.values():
            assert len(metric) == self.n

        return metrics

    def plot_and_save_result(self, output_path: str) -> None:
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:cyan', 'tab:olive', 'tab:orange', 'tab:purple']
        figure = plt.figure(figsize=(16, 10), dpi=80)
        for ind, i in enumerate(self.metrics):
            plt.plot(self.metrics[i], color=colors[ind], label=i)

        plt.xlim([0, (self.n-1)])
        plt.xlabel(r'$Итерации$', fontsize=16)
        plt.ylabel(r'$Метрики$', fontsize=16)
        plt.legend(fontsize=14)

        plt.grid()
        figure.savefig(output_path)

        return plt.show()


# a = Probs_algo(data_path='D:/P-projects/test/test.csv', probs=[0.5, 0.25, 0.25], n=10, labels=[0, 1, 2])
# a.plot_and_save_result('tt')