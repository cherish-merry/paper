import itertools
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, log_loss, confusion_matrix
import xgboost as xgb
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import matplotlib
matplotlib.rcParams.update({'font.size': 14})  #
plt.rcParams['font.family'] = 'Arial Unicode MS'  # 设置字体为 Arial Unicode MS
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置使用指定字体

class SoftDecisionTree:
    def __init__(self, max_depth=None, min_sample_leaf=None):
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.nodes = 0

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):

        classes = np.argmax(y, axis=1)
        unique_classes = np.unique(classes)
        # 如果所有样本属于同一类别，创建叶节点
        if len(unique_classes) == 1:
            self.nodes += 1
            return self._generate_node(y)

        # 如果达到最大深度，创建叶节点，以多数票决定类别
        if self.max_depth != None and depth == self.max_depth:
            self.nodes += 1
            return self._generate_node(y)

        if self.min_sample_leaf != None and len(X) <= self.min_sample_leaf:
            self.nodes += 1
            return self._generate_node(y)

        # 选择最佳的特征和切分点
        best_feature, best_value = self._find_best_split(X, y)

        # 如果无法找到合适的切分点，创建叶节点，以多数票决定类别
        if best_feature is None:
            self.nodes += 1
            return self._generate_node(y)

        # 递归地创建左右子树
        left_indices = X[:, best_feature] <= best_value
        right_indices = ~left_indices

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': best_feature, 'value': best_value,
                'left': left_subtree, 'right': right_subtree}

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None  # 无法进行切分

        # 计算当前节点的基尼系数
        best_gini = self._calculate_soft_gini(y)
        best_feature = None
        best_value = None

        for feature in range(n_features):
            # 获取特征列
            feature_values = X[:, feature]

            # 尝试不同的切分点
            for value in np.unique(feature_values):
                left_indices = feature_values <= value
                right_indices = ~left_indices

                # 计算切分后的基尼系数
                left_gini = self._calculate_soft_gini(y[left_indices])
                right_gini = self._calculate_soft_gini(y[right_indices])

                # 计算加权基尼系数
                weighted_gini = (len(y[left_indices]) / n_samples) * left_gini + \
                                (len(y[right_indices]) / n_samples) * right_gini

                # 更新最佳切分点
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_value = value

        return best_feature, best_value

    def _calculate_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini

    def _calculate_soft_gini(self, y):
        assert np.ndim(y) == 2
        if len(y) == 0: return 0
        y = np.array(y)
        sum = 0
        for i in range(y.shape[1]):
            sum += (np.sum(y[:, i]) / y.shape[0]) ** 2
        return 1 - sum

    def _generate_node(self, y):
        y_proba = np.mean(y, axis=0)
        majority_class = np.argmax(y_proba)
        return {'class': majority_class, 'proba': y_proba}

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def predict_proba(self, X):
        return np.array([self._predict_tree_proba(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        if 'class' in node:
            return node['class']
        if x[node['feature']] <= node['value']:
            return self._predict_tree(x, node['left'])
        else:
            return self._predict_tree(x, node['right'])

    def _predict_tree_proba(self, x, node):
        if 'class' in node:
            return node['proba']
        if x[node['feature']] <= node['value']:
            return self._predict_tree_proba(x, node['left'])
        else:
            return self._predict_tree_proba(x, node['right'])


class DecisionTree:
    def __init__(self, max_depth=None, min_sample_leaf=None):
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.nodes = 0

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        unique_classes = np.unique(y)

        # 如果所有样本属于同一类别，创建叶节点
        if len(unique_classes) == 1:
            self.nodes += 1
            return {'class': unique_classes[0]}

        # 如果达到最大深度，创建叶节点，以多数票决定类别
        if self.max_depth != None and depth == self.max_depth:
            majority_class = np.bincount(y).argmax()
            self.nodes += 1
            return {'class': majority_class}

        if self.min_sample_leaf != None and len(X) <= self.min_sample_leaf:
            majority_class = np.bincount(y).argmax()
            self.nodes += 1
            return {'class': majority_class}

        # 选择最佳的特征和切分点
        best_feature, best_value = self._find_best_split(X, y)

        # 如果无法找到合适的切分点，创建叶节点，以多数票决定类别
        if best_feature is None:
            majority_class = np.bincount(y).argmax()
            self.nodes += 1
            return {'class': majority_class}

        # 递归地创建左右子树
        left_indices = X[:, best_feature] <= best_value
        right_indices = ~left_indices

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': best_feature, 'value': best_value,
                'left': left_subtree, 'right': right_subtree}

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None  # 无法进行切分

        # 计算当前节点的基尼系数
        best_gini = self._calculate_gini(y)
        best_feature = None
        best_value = None

        for feature in range(n_features):
            # 获取特征列
            feature_values = X[:, feature]

            # 尝试不同的切分点
            for value in np.unique(feature_values):
                left_indices = feature_values <= value
                right_indices = ~left_indices

                # 计算切分后的基尼系数
                left_gini = self._calculate_gini(y[left_indices])
                right_gini = self._calculate_gini(y[right_indices])

                # 计算加权基尼系数
                weighted_gini = (len(y[left_indices]) / n_samples) * left_gini + \
                                (len(y[right_indices]) / n_samples) * right_gini

                # 更新最佳切分点
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_value = value

        return best_feature, best_value

    def _calculate_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        if 'class' in node:
            return node['class']
        if x[node['feature']] <= node['value']:
            return self._predict_tree(x, node['left'])
        else:
            return self._predict_tree(x, node['right'])


def load_data(frac):
    df = pd.read_csv("sample-5w.csv").sample(frac=frac, random_state=42)
    df['Label'], mapping = pd.factorize(df['Label'])
    df = df.applymap(lambda x: np.floor(x))
    train, test = train_test_split(df, test_size=0.2, random_state=0)
    print(mapping)
    return mapping, train.values, test.values


def produce_soft_labels(data, round_num, fold_num, model_name, k, T):
    soft_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    for i in range(round_num):
        kf = KFold(n_splits=fold_num)
        for train_index, test_index in kf.split(X=data[:, :-1], y=data[:, -1], groups=data[:, -1]):
            train_set, test_set = data[train_index], data[test_index]
            train_X, train_Y = train_set[:, :-1], train_set[:, -1].astype(int)
            test_X = test_set[:, :-1]
            clf = xgb.XGBClassifier()
            clf.fit(train_X, train_Y)
            raw_scores = clf.predict(test_X, output_margin=True)
            raw_scores_tensor = torch.tensor(raw_scores)
            pred_prob = F.softmax(raw_scores_tensor / T, dim=1).detach().numpy()
            # rf.fit(train_X, train_Y)
            # pred_prob = clf.predict_proba(test_X)
            soft_label[test_index] += pred_prob

    soft_label /= round_num
    hard_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    for i in range(np.shape(data)[0]):
        hard_label[i][int(data[i, -1])] = 1
    soft_label = (soft_label + hard_label * k) / (k + 1)
    # return hard_label
    return soft_label


def evaluate(model_name, k, frac, fold, min_sample_leaf, max_depth, T):
    target_names, data_train, data_eval = load_data(frac=frac)
    soft_label = produce_soft_labels(data_train, 1, fold, model_name, k, T)
    clf = SoftDecisionTree(max_depth=max_depth, min_sample_leaf=min_sample_leaf)
    start_time = time.time()
    clf.fit(data_train[:, :-1], soft_label)
    end_time = time.time()
    print("SDT 训练时间：", end_time - start_time, "秒")
    y_pred = clf.predict(data_eval[:, :-1])
    y_pred_proba = clf.predict_proba(data_eval[:, :-1])
    student_report = classification_report(data_eval[:, -1], y_pred, target_names=target_names, zero_division=1,
                                           digits=3)

    y_pred_train = clf.predict(data_train[:, :-1])
    student_report_train = classification_report(data_train[:, -1], y_pred_train, target_names=target_names,
                                                 zero_division=1,
                                                 digits=3)

    # print("studnet train")
    # print(student_report_train)

    export_cm(data_eval[:, -1], y_pred)

    th = xgb.XGBClassifier()
    th.fit(data_train[:, :-1], data_train[:, -1])
    pred = th.predict(data_eval[:, :-1])
    teacher_report = classification_report(data_eval[:, -1], pred, target_names=target_names, zero_division=1,
                                           digits=3)

    # 普通决策树
    dt = DecisionTree(max_depth=max_depth)
    clf.fit(data_train[:, :-1], soft_label)
    dt.fit(data_train[:, :-1], data_train[:, -1].astype(int))
    end_time = time.time()
    print("DT 训练时间：", end_time - start_time, "秒")
    rf_pred = dt.predict(data_eval[:, :-1])

    origin_report = classification_report(data_eval[:, -1], rf_pred, target_names=target_names, zero_division=1,
                                          digits=3)

    rf_pred_train = dt.predict(data_train[:, :-1])

    origin_report_train = classification_report(data_train[:, -1], rf_pred_train, target_names=target_names,
                                                zero_division=1,
                                                digits=3)

    loss = log_loss(data_eval[:, -1], y_pred_proba)

    print("teacher")
    print(teacher_report)

    print("studnet")
    print(student_report)

    print("origin_train")
    print(origin_report_train)

    print("origin")
    print(origin_report)

    data = {
        "model_name": model_name,
        "k": k,
        "frac": frac,
        "T": int(T),
        "student_nodes": clf.nodes,
        "origin_nodes": dt.nodes,
        "min_sample_leaf": min_sample_leaf,
        "max_depth": max_depth,
        "log_loss": loss,
        "student_report": student_report,
        "teacher_report": teacher_report,
        "origin_report": origin_report,
    }

    print(data)
    return data


def export_cm(y_true, y_pred):
    labels = [0, 1, 2, 3, 4, 5]
    target_names = ["BENIGN", "DoS Slowloris", "DoS Hulk", "DoS Slowhttptest", "DoS GoldenEye", "DDoS LOIT"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    # plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')  # 调整x轴标签角度为45度，并靠右对齐
    plt.yticks(rotation=45, va='center')  # 调整y轴标签角度为45度，并居中对齐
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig("confusion_matrix_xgb.png", dpi=500, format="png")
    # plt.show()


def write_result(filename, result):
    with open(filename, 'a') as file:
        file.write(result)  # 添加一个空行以区分不同的报告


if __name__ == "__main__":
    result = []

    model_names = ["xgb"]
    T = [1]
    fold = [3]
    ks = [0]
    fracs = [1]
    min_sample_leafs = [None]
    max_depths = [None]
    parameter_space = list(itertools.product(model_names, ks, fracs, fold, min_sample_leafs, max_depths, T))

    # 遍历参数空间，并传递参数给evaluate函数
    for params in tqdm(parameter_space, desc="Parameter Space"):
        data = evaluate(*params)
        result.append(data)

    with open('xgb_test.json', 'w') as f:
        json.dump(result, f, indent=4)
