from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import (
    compute_confident_joint,
    estimate_latent,
)
import copy
import numpy as np
from ml_datatset import get_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

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

def get_noise(xall_new, label_new, pre_new, all_nodes, all_fdirs, all_labels):
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
    x_pruned = xall_new1[x_mask]
    s_pruned = y_train2[x_mask]

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


    # print(new_fdirs.shape)
    # print(new_nodes)

    # print(all_fdirs.shape)
    # print("--------------------------")
    # print(new_fdirs.shape)
    #
    # print(new_nodes)
    # print(fDirMap)
    # print(x_mask)
    return x_mask


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






    # all_proba_pairs = clf.predict_proba(test_x).tolist()
    # dir_to_minigraphs = {}
    # eval(test_all_fdirs, dir_to_minigraphs, all_proba_pairs, test_all_nodes)
    #
    # svm_rtop1.append(eval_recall_topk(test_data, dir_to_minigraphs, 1))
    # svm_rtop2.append(eval_recall_topk(test_data, dir_to_minigraphs, 2))
    # svm_rtop3.append(eval_recall_topk(test_data, dir_to_minigraphs, 3))
    # svm_mfr.append(eval_mean_first_rank(test_data, dir_to_minigraphs))