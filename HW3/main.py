import pandas as pd
from loguru import logger
import random
import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import preprocess, plot_learners_roc, accuracy_score
from src.decision_tree import gini, entropy
import matplotlib.pyplot as plt


def main():
    """
    Note:
    1) Part of line should not be modified.
    2) You should implement the algorithm by yourself.
    3) You can change the I/O data type as you need.
    4) You can change the hyperparameters as you want.
    5) You can add/modify/remove args in the function, but you need to fit the requirements.
    6) When plot the feature importance, the tick labels of one of the axis should be feature names.
    """
    random.seed(777)  # DON'T CHANGE THIS LINE
    torch.manual_seed(777)  # DON'T CHANGE THIS LINE
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()

    # (TODO): Implement you preprocessing function.
    features = X_train.columns
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    y_train = torch.tensor(y_train, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    print(f"finish data preprocessing, len(y_test) = {len(y_test)}")

    """
    (TODO): Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """
    # AdaBoost
    print("adaboost")
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=500,
        learning_rate=0.001,
    )
    # (500, 0.001) = 0.7924
    # (1000, 0.001) = 0.7980
    # (8000, 0.0009) = 0.8067
    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="img/adaboost_roc.png",
    )
    feature_importance = clf_adaboost.compute_feature_importance()
    # (TODO) Draw the feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(features, feature_importance, color='#6495ED', label='importance')
    plt.legend(loc='lower right')
    plt.savefig("img/adaboost_importance.png")

    # Bagging
    print("bagging")
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=500,
        learning_rate=0.001,
    )
    # (500, 0.001) = 0.7192
    # (1000, 0.001) = 0.7778 -> good roc curve
    # (1000, 0.001) = 0.7798 (pred_class > n_learner/2)
    # (8000, 0.0009) = 0.7778 -> bad roc curve
    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="img/bagging_roc.png",
    )
    feature_importance = clf_bagging.compute_feature_importance()
    # (TODO) Draw the feature importance
    # features = X_train.columns.tolist()
    plt.figure(figsize=(10, 8))
    plt.barh(features, feature_importance, color='#6495ED', label='importance')
    plt.legend(loc='lower right')
    plt.savefig("img/bagging_importance.png")

    # Decision Tree
    print("decision tree")
    a = [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]
    print(f"gini index of the array = {gini(a)}")
    print(f"entropy of the array = {entropy(a)}")
    clf_tree = DecisionTree(
        max_depth=7,
    )
    clf_tree.fit(X_train.numpy(), y_train.numpy())
    y_pred_classes = clf_tree.predict(X_test.numpy())
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')
    # plot feature improtance
    plt.figure(figsize=(10, 8))
    feature_importance = clf_tree.compute_feature_importance()
    plt.barh(features, feature_importance, color='#6495ED', label='importance')
    plt.legend(loc='lower right')
    plt.savefig("img/tree_importance.png")


if __name__ == '__main__':
    main()
