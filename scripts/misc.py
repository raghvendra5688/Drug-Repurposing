import os
import pickle
import numpy as np
from sklearn import metrics
from sklearn import utils
from sklearn import model_selection
import scipy
import matplotlib.pyplot as plt


def save_model(model, filename):

    outpath = os.path.join("../models/", filename)

    with open(outpath, "wb") as f:
        pickle.dump(model, f)

    print("Saved model to file: %s" % (outpath))


def load_model(filename):

    fpath = os.path.join("../models/", filename)

    with open(fpath, "rb") as f:
        model = pickle.load(f)

    print("Load model to file: %s" % (fpath))
    return model


def classification_results(y_true, y_pred, normalize=False, title=None,  cmap=plt.cm.Blues, class_names=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print("-" * 80)
    print(metrics.classification_report(y_true, y_pred))
    
    if title is None:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data

    class_labels = utils.multiclass.unique_labels(y_true, y_pred)
    
    if class_names is None:
        classes = class_labels
    elif len(class_names) == len(class_labels):
        classes = class_names
    else:
        print("ERROR: Found %d classes, but got a list with only %d classes (%s)" % (len(class_labels), len(class_names), class_names))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label', )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    color="white" if cm[i, j] > thresh else "black")
    plt.show()
    print("=" * 80)
    return ax

def regression_results(model, y_true, y_pred):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print("-" * 80)
    print("Model: %s" % (model))
    print("-" * 80)
    results = []
    for metric in [metrics.mean_squared_error, metrics.mean_squared_log_error, metrics.mean_absolute_error,
                   metrics.explained_variance_score, metrics.median_absolute_error, metrics.r2_score]:

        res = metric(y_true, y_pred)
        results.append(res)
        print("%s: %.3f" % (metric.__name__, res))
    res = scipy.stats.pearsonr(np.array(y_true),np.array(y_pred))[0]
    results.append(res)
    print("Pearson R: %.3f" %(res))

    print("=" * 80)
    return results

def grid_search_cv(model, parameters, X_train, y_train, n_splits=5, n_iter=1000, n_jobs=-1, scoring="r2", stratified=False):
    """
        Tries all possible values of parameters and returns the best regressor/classifier.
        Cross Validation done is stratified.

        See scoring options at https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    """

    # Stratified n_splits Folds. Shuffle is not needed as X and Y were already shuffled before.
    if stratified:
        cv = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=42)
    else:
        cv = n_splits

    model = model_selection.RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=cv, scoring=scoring, n_iter=n_iter, n_jobs=n_jobs,random_state=0,
                                         verbose=2)
    return model.fit(X_train, y_train)
