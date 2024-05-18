import itertools
import json

import numpy as np
import pandas as pd
import torch
from matplotlib import gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, log_loss, confusion_matrix
import xgboost as xgb
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
from mlp import MLP
import seaborn as sns
import matplotlib
# matplotlib.rcParams.update({'font.size': 12})  #
# plt.rcParams['font.family'] = 'Arial Unicode MS'  # 设置字体为 Arial Unicode MS
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置使用指定字体



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
    train, test = train_test_split(df, test_size=0.5, random_state=0)
    return mapping, train.values, test.values


def produce_xgb_soft_labels(data, round_num, fold_num, k):
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
            pred_prob = F.softmax(raw_scores_tensor, dim=1).detach().numpy()
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


def produce_mlp_soft_labels(data, round_num, k=0.5):
    soft_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    for i in range(round_num):
        train_X, train_Y = data[:, :-1], data[:, -1].astype(int)
        clf = MLP(train_X.shape[1], 6)
        clf.fit(train_X, train_Y)
        pred_prob = clf.predict_proba(train_X, 1)

        soft_label += pred_prob

    soft_label /= round_num
    hard_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    for i in range(np.shape(data)[0]):
        hard_label[i][int(data[i, -1])] = 1
    soft_label = (soft_label + hard_label * k) / (k + 1)

    return soft_label


def evaluate(k, frac, fold, max_depth):
    target_names, data_train, data_eval = load_data(frac=frac)
    soft_label_xgb = produce_xgb_soft_labels(data_train, 1, fold, k)
    soft_label_mlp = produce_mlp_soft_labels(data_train, 2, k)
    # 优化后的决策树
    clf_xgb = SoftDecisionTree(max_depth=max_depth)
    clf_xgb.fit(data_train[:, :-1], soft_label_xgb)
    sdt_xgb_pred = clf_xgb.predict(data_eval[:, :-1])
    student_xgb_report = classification_report(data_eval[:, -1], sdt_xgb_pred, target_names=target_names,
                                               zero_division=1,
                                               digits=3)

    th = xgb.XGBClassifier()
    th.fit(data_train[:, :-1], data_train[:, -1])
    xgb_pred = th.predict(data_eval[:, :-1])
    teacher_xgb_report = classification_report(data_eval[:, -1], xgb_pred, target_names=target_names, zero_division=1,
                                               digits=3)

    clf_mlp = SoftDecisionTree(max_depth=max_depth)
    clf_mlp.fit(data_train[:, :-1], soft_label_mlp)
    sdt_mlp_pred = clf_mlp.predict(data_eval[:, :-1])
    student_mlp_report = classification_report(data_eval[:, -1], sdt_mlp_pred, target_names=target_names,
                                               zero_division=1,
                                               digits=3)

    train_X, train_Y = data_train[:, :-1], data_train[:, -1]
    th = MLP(train_X.shape[1], len(target_names))
    th.fit(data_train[:, :-1], data_train[:, -1])
    mlp_pred = th.predict(data_eval[:, :-1])
    teacher_mlp_report = classification_report(data_eval[:, -1], mlp_pred, target_names=target_names, zero_division=1,
                                               digits=3)

    # 普通决策树
    dt = DecisionTree(max_depth=max_depth)
    dt.fit(data_train[:, :-1], data_train[:, -1].astype(int))
    dt_pred = dt.predict(data_eval[:, :-1])
    origin_report = classification_report(data_eval[:, -1], dt_pred, target_names=target_names, zero_division=1,
                                          digits=3)
    # print("teacher")
    # print(teacher_report)

    # print("studnet")
    # print(student_report)
    #
    # print("origin_train")
    # print(origin_report_train)
    #
    # print("origin")
    # print(origin_report)
    plot(data_eval[:, -1], dt_pred, sdt_xgb_pred, xgb_pred, sdt_mlp_pred, mlp_pred)


def plot(y_true, y_pred_A, y_pred_B, y_pred_C, y_pred_D, y_pred_E):
    # 找到模型A和模型B预测结果不同的索引
    # fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    fig = plt.figure(tight_layout=True, figsize=(9, 3))
    gs = gridspec.GridSpec(1, 3)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 0])
    ax5 = fig.add_subplot(gs[0, 1])
    ax6 = fig.add_subplot(gs[0, 2])
    # ax1.set_title("(a) DT")
    # ax2.set_title("(b) DT-XGB")
    # ax3.set_title("(c) XGB")
    ax4.set_title("(d) DT")
    ax5.set_title("(e) DT-MLP")
    ax6.set_title("(f) MLP")
    classes = [0, 1, 2, 3, 4, 5]

    diff_indices = [i for i in range(len(y_true)) if y_pred_A[i] != y_pred_B[i]]

    # 筛选出有差异的样本
    y_true_diff = [y_true[i] for i in diff_indices]
    y_pred_A_diff = [y_pred_A[i] for i in diff_indices]
    y_pred_B_diff = [y_pred_B[i] for i in diff_indices]
    y_pred_C_diff = [y_pred_C[i] for i in diff_indices]

    # 计算混淆矩阵

    cm_model_A = fit_cm(y_true_diff, y_pred_A_diff)
    cm_model_B = fit_cm(y_true_diff, y_pred_B_diff)
    cm_model_C = fit_cm(y_true_diff, y_pred_C_diff)

    diff_indices = [i for i in range(len(y_true)) if y_pred_A[i] != y_pred_D[i]]
    # 筛选出有差异的样本
    y_true_diff = [y_true[i] for i in diff_indices]
    y_pred_A_diff = [y_pred_A[i] for i in diff_indices]
    y_pred_D_diff = [y_pred_D[i] for i in diff_indices]
    y_pred_E_diff = [y_pred_E[i] for i in diff_indices]

    # 计算混淆矩阵
    cm_model_A2 = fit_cm(y_true_diff, y_pred_A_diff)
    cm_model_D = fit_cm(y_true_diff, y_pred_D_diff)
    cm_model_E = fit_cm(y_true_diff, y_pred_E_diff)

    x = 0

    # sns.heatmap(cm_model_A, ax=ax1, cmap='Blues', fmt='d', cbar=False, annot=True)
    # sns.heatmap(cm_model_B, ax=ax2, cmap='Blues', fmt='d', cbar=False, annot=True)
    # sns.heatmap(cm_model_C, ax=ax3, cmap='Blues', fmt='d', cbar=False, annot=True)
    sns.heatmap(cm_model_A2, ax=ax4, cmap='Blues', fmt='d', cbar=False, annot=True)
    sns.heatmap(cm_model_D, ax=ax5, cmap='Blues', fmt='d', cbar=False, annot=True)
    sns.heatmap(cm_model_E, ax=ax6, cmap='Blues', fmt='d', cbar=False, annot=True)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # 创建水平方向的颜色条并共享
    # cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # 定义颜色条的位置，右边
    # fig.colorbar(ax1.collections[0], cax=cbar_ax)
    plt.savefig("diff9.png", dpi=500, format="png")
    # plt.show()


def fit_cm(y_true, y_pred):
    labels = [0, 1, 2, 3, 4, 5]
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[labels.index(true), labels.index(pred)] += 1
    return cm


if __name__ == "__main__":
    fold = [3]
    ks = [0.5]
    fracs = [1]
    max_depths = [None]
    parameter_space = list(itertools.product(ks, fracs, fold, max_depths))

    # 遍历参数空间，并传递参数给evaluate函数
    for params in tqdm(parameter_space, desc="Parameter Space"):
        data = evaluate(*params)
