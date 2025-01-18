from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import (
    compute_confident_joint,
    estimate_latent,
)
import sys
sys.path.append('replication/noise_baseline/')
sys.path.append('gbs_src/')
sys.path.append('e2sc_src')
import copy
from noise_baseline import RSDS
from noise_baseline import e2sc
from noise_baseline.gbs_src import GBS

import numpy as np
from imblearn.under_sampling import OneSidedSelection
from sklearn.ensemble import IsolationForest
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer
from ml_datatset import get_dataset


def get_graph(all_fdirs, all_nodes):
    fDirMap = {}
    graph = []
    for i in range(len(all_fdirs)):
        if i == 0 or all_fdirs[i] != all_fdirs[i-1]:
            graph = []

        graph.append(all_nodes[i])

        if i+1 >= len(all_fdirs) or all_fdirs[i] != all_fdirs[i+1]:
            fDirMap[all_fdirs[i]] = graph

    return fDirMap


def selectFromLinearSVC2(train_content, train_label):
    lsvc = LinearSVC(max_iter=50000).fit(train_content, train_label)
    model = SelectFromModel(lsvc, prefit=True)

    new_train = model.transform(train_content)

    return new_train



def get_noise_cl(xall_new, label_new, pre_new, all_nodes, all_fdirs, all_labels):
    xall_new1 = copy.deepcopy(xall_new)
    label_new1 = copy.deepcopy(label_new)
    pre_new1 = copy.deepcopy(pre_new)

    label_1 = label_new1.ravel()
    y_train2 = label_1.astype(np.int16)
    confident_joint = compute_confident_joint(
        s=y_train2,
        psx=pre_new1,  # P(s = k|x)
        thresholds=None
    )

    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=y_train2,
        py_method='cnt',
        converge_latent_estimates=False
    )

    ordered_label_errors = get_noise_indices(
        s=y_train2,
        psx=pre_new1,
        inverse_noise_matrix=inv_noise_matrix,
        confident_joint=confident_joint,
        prune_method='prune_by_noise_rate',
    )

    # print(ordered_label_errors)

    x_mask = ~ordered_label_errors
    all_fdirs = np.array(all_fdirs)
    all_nodes = np.array(all_nodes)
    all_labels = np.array(all_labels)

    print("数组中正样本个数", np.sum(all_labels == 1))
    print("数组中负样本个数", np.sum(all_labels == 0))

    new_fdirs = all_fdirs[x_mask]
    new_nodes = all_nodes[x_mask]
    new_labels = all_labels[x_mask]

    print("过滤之后的数组中正样本个数", np.sum(new_labels == 1))
    print("过滤之后的数组中负样本个数", np.sum(new_labels == 0))
    return x_mask


# - CLNI方法
def get_noise_clni(xall_new, label_new, pre_new, all_nodes, all_fdirs, all_labels):
    xall_new1 = copy.deepcopy(xall_new)
    label_new1 = copy.deepcopy(label_new)
    pre_new1 = copy.deepcopy(pre_new)

    label_1 = label_new1.ravel()
    y_train2 = label_1.astype(np.int16)
    ordered_label_errors = []

    count_no = 0
    num_samples = len(label_new1)
    for ii in range(num_samples):
        r_pre1 = np.zeros((num_samples, 2))
        for jj in range(num_samples):
            r_pre1[jj][0] = np.sqrt(np.sum((xall_new1[ii] - xall_new1[jj]) ** 2))  # 计算欧氏距离
            r_pre1[jj][1] = label_new1[jj]

        idex = np.argsort(r_pre1[:, 0])
        sorted_data = r_pre1[idex, :]

        countc = 0
        for jj in range(1, 6):  # 跳过自己，最近邻从索引 1 开始
            if label_new1[ii] != sorted_data[jj][1]:
                countc += 1
        the_1 = countc / 5  # 最近 5 个邻居中标签不同的比例

        # 判断噪声样本
        if the_1 >= 0.6:
            ordered_label_errors.append(False)  # 认为是噪声
            count_no += 1
        else:
            ordered_label_errors.append(True)  # 非噪声

        if count_no > num_samples * 0.01:
            break

    if len(ordered_label_errors) < num_samples:
        ordered_label_errors.extend([True] * (num_samples - len(ordered_label_errors)))

    ordered_label_errors = np.array(ordered_label_errors)
    x_mask = ordered_label_errors  # 过滤噪声样本

    all_fdirs = np.array(all_fdirs)
    all_nodes = np.array(all_nodes)
    all_labels = np.array(all_labels)

    print("数组中正样本个数", np.sum(all_labels == 1))
    print("数组中负样本个数", np.sum(all_labels == 0))

    new_fdirs = all_fdirs[x_mask]
    new_nodes = all_nodes[x_mask]
    new_labels = all_labels[x_mask]

    print("过滤之后的数组中正样本个数", np.sum(new_labels == 1))
    print("过滤之后的数组中负样本个数", np.sum(new_labels == 0))
    print(len(x_mask))
    return x_mask

def get_noise_rsds(xall_new, label_new, pre_new, all_nodes, all_fdirs, all_labels):
    # 使用 rsds
    xall_new1 = copy.deepcopy(xall_new)
    label_new1 = copy.deepcopy(label_new)
    oss = OneSidedSelection(random_state=42, n_jobs=-1)
    x_resampled = RSDS.RSDS_fun(xall_new,)

    # 创建掩码
    x_mask = np.isin(np.arange(len(xall_new)), np.arange(len(x_resampled)))

    all_fdirs = np.array(all_fdirs)
    all_nodes = np.array(all_nodes)
    all_labels = np.array(all_labels)

    print("数组中正样本个数", np.sum(all_labels == 1))
    print("数组中负样本个数", np.sum(all_labels == 0))

    new_fdirs = all_fdirs[x_mask]
    new_nodes = all_nodes[x_mask]
    new_labels = all_labels[x_mask]

    print("过滤之后的数组中正样本个数", np.sum(new_labels == 1))
    print("过滤之后的数组中负样本个数", np.sum(new_labels == 0))
    return x_mask

# 使用Isolation Forest（IF）方法进行噪声检测
def get_noise_IF(xall_new, label_new, pre_new, all_nodes, all_fdirs, all_labels):
    xall_new1 = copy.deepcopy(xall_new)
    label_new1 = copy.deepcopy(label_new)
    pre_new1 = copy.deepcopy(pre_new)

    if_model = IsolationForest(contamination='auto',random_state=12)
    if_model.fit(xall_new1)
    noise_labels = if_model.predict(xall_new1)

    x_mask = noise_labels
    x_mask = np.where(x_mask==-1,1,0)
    x_mask = np.array(x_mask,dtype=bool)
    x_mask = ~x_mask


    all_fdirs = np.array(all_fdirs)
    all_nodes = np.array(all_nodes)
    all_labels = np.array(all_labels)

    print("数组中正样本个数", np.sum(all_labels == 1))
    print("数组中负样本个数", np.sum(all_labels == 0))

    new_fdirs = all_fdirs[x_mask]
    new_nodes = all_nodes[x_mask]
    new_labels = all_labels[x_mask]

    print("过滤之后的数组中正样本个数", np.sum(new_labels == 1))
    print("过滤之后的数组中负样本个数", np.sum(new_labels == 0))
    return x_mask

def get_noise_gbs(xall_new, label_new, pre_new, all_nodes, all_fdirs, all_labels):
    # 使用 gbs进行噪声过滤
    xall_new1 = copy.deepcopy(xall_new)
    label_new1 = copy.deepcopy(label_new)

    x_resampled, y_resampled = GBS.main(xall_new1, label_new1.ravel())

    # 创建掩码
    x_mask = np.isin(np.arange(len(xall_new)), np.arange(len(x_resampled)))

    all_fdirs = np.array(all_fdirs)
    all_nodes = np.array(all_nodes)
    all_labels = np.array(all_labels)

    print("数组中正样本个数", np.sum(all_labels == 1))
    print("数组中负样本个数", np.sum(all_labels == 0))

    new_fdirs = all_fdirs[x_mask]
    new_nodes = all_nodes[x_mask]
    new_labels = all_labels[x_mask]

    print("过滤之后的数组中正样本个数", np.sum(new_labels == 1))
    print("过滤之后的数组中负样本个数", np.sum(new_labels == 0))
    return x_mask


def get_noise_oss(xall_new, label_new, pre_new, all_nodes, all_fdirs, all_labels):
    # 使用 One-Sided Selection (OSS) 进行噪声过滤
    xall_new1 = copy.deepcopy(xall_new)
    label_new1 = copy.deepcopy(label_new)
    oss = OneSidedSelection(random_state=42, n_jobs=-1)
    x_resampled, y_resampled = oss.fit_resample(xall_new1, label_new1.ravel())

    # 创建掩码
    x_mask = np.isin(np.arange(len(xall_new)), np.arange(len(x_resampled)))

    all_fdirs = np.array(all_fdirs)
    all_nodes = np.array(all_nodes)
    all_labels = np.array(all_labels)

    print("数组中正样本个数", np.sum(all_labels == 1))
    print("数组中负样本个数", np.sum(all_labels == 0))

    new_fdirs = all_fdirs[x_mask]
    new_nodes = all_nodes[x_mask]
    new_labels = all_labels[x_mask]

    print("过滤之后的数组中正样本个数", np.sum(new_labels == 1))
    print("过滤之后的数组中负样本个数", np.sum(new_labels == 0))
    return x_mask

def get_noise_e2sc(xall_new, label_new, pre_new, all_nodes, all_fdirs, all_labels):
    xall_new1 = copy.deepcopy(xall_new)
    label_new1 = copy.deepcopy(label_new)
    selector = e2sc.E2SC()
    selector.select_data(xall_new1,label_new1)


    # 创建掩码
    x_mask = np.isin(np.arange(len(xall_new)), np.arange(len(selector.sample_indices_)))

    all_fdirs = np.array(all_fdirs)
    all_nodes = np.array(all_nodes)
    all_labels = np.array(all_labels)

    print("数组中正样本个数", np.sum(all_labels == 1))
    print("数组中负样本个数", np.sum(all_labels == 0))

    new_fdirs = all_fdirs[x_mask]
    new_nodes = all_nodes[x_mask]
    new_labels = all_labels[x_mask]

    print("过滤之后的数组中正样本个数", np.sum(new_labels == 1))
    print("过滤之后的数组中负样本个数", np.sum(new_labels == 0))
    return x_mask

def get_noise(xall_new, label_new, pre_new, all_nodes, all_fdirs, all_labels, method):
    methods = {
        'cl':get_noise_cl,
        'clni': get_noise_clni,
        'if': get_noise_IF,
        'e2sc':get_noise_e2sc,
        'oss':get_noise_oss,
        'rsds':get_noise_rsds,
        'gbs':get_noise_gbs
    }

    if method not in methods:
        raise ValueError(f"Method {method} is not supported.")

    return methods[method](xall_new, label_new, pre_new, all_nodes, all_fdirs, all_labels)


def get_vectorizedata(data):

    (
        train_all_nodes,
        train_all_codes,
        train_all_labels,
        train_all_fdirs,
        train_all_info,
    ) = get_dataset(data)

    # print(train_all_codes)
    vectorizer = CountVectorizer(max_features=10000, tokenizer=None, stop_words=None)

    train_content_matrix = vectorizer.fit_transform(train_all_codes)

    # print(train_content_matrix.shape)

    train_content_matrix = selectFromLinearSVC2(
        train_content_matrix, train_all_labels
    )

    train_x = train_content_matrix.toarray()
    train_y = np.array(train_all_labels)

    # train_x = train_x[:100, ]
    # train_y = train_y[:100, ]
    # train_all_nodes = train_all_nodes[:100]
    # train_all_fdirs = train_all_fdirs[:100]

    # print(train_x.shape)
    # print(train_y.shape)
    # print("---------")

    psx = np.zeros((len(train_y), 2))

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(train_x, train_y)):
        # print(cv_train_idx)
        # Select the training and holdout cross-validated sets.
        X_train_cv, X_holdout_cv = train_x[cv_train_idx], train_x[cv_holdout_idx]
        s_train_cv, s_holdout_cv = train_y[cv_train_idx], train_y[cv_holdout_idx]

        clf = SVC(kernel="rbf", probability=True, random_state=12)
        clf.fit(X_train_cv, s_train_cv)
        psx_cv = clf.predict_proba(X_holdout_cv)
        psx[cv_holdout_idx] = psx_cv


    return train_x, train_y, psx, train_all_nodes, train_all_fdirs, train_all_labels