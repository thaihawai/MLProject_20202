import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix

from keras import models
from gensim.models.doc2vec import Doc2Vec


def format_length(arr, length, vocab):
    while len(arr) < length:
        arr.append(vocab['<pad>'])
    newArr = arr[:(length-1)]
    if arr[length-1] == vocab['<pad>']:
        newArr.append(vocab['<pad>'])
    else:
        newArr.append(vocab['<end>'])
    return newArr


def get_lstm_vectors(lstm_model, MODEL_INPUT, MODEL_OUTPUT, embedded_text):
    # i.e 256 output from lstm layers, 8 input
    # -> INPUT = 8, OUTPUT = 256
    context_vectors = np.empty((0, MODEL_OUTPUT))
    # start sequence is 1 with all zeros behind
    start_sequence = np.zeros((1, MODEL_INPUT))
    start_sequence[0][0] = 1
    # index of lstm model = 8 because we use 2 lstm layers
    feature_extractor = models.Model(inputs = lstm_model.inputs, outputs = lstm_model.get_layer(index=8).output)
    for doc in embedded_text:
        vector_to_context = np.reshape(np.array(doc), (-1, len(doc)))
        features = feature_extractor([vector_to_context, start_sequence])
        context_vectors = np.append(context_vectors, np.array(features), axis=0)
    return context_vectors


def get_doc2vec_vectors(doc2vec_model, MODEL_OUTPUT, text_all):
    # output either 20 or 100 depend on model used
    context_vectors = np.empty((0, MODEL_OUTPUT))
    # text_all = list of input
    for doc in text_all:
        features = doc2vec_model.infer_vector(doc)
        context_vectors = np.append(context_vectors, np.expand_dims(features, axis=0), axis=0)
    return context_vectors


def plot_loss(history, label, n, colors):
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
            color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
            color=colors[n], label='Val ' + label,
            linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')


def plot_metrics(history, colors):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
            color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('True Job Postings Detected (True Negatives): ', cm[0][0])
    print('True Job Postings Incorrectly Detected (False Positives): ', cm[0][1])
    print('False Job Postings Missed (False Negatives): ', cm[1][0])
    print('False Job Postings Detected (True Positives): ', cm[1][1])
    print('Total False Job Postings: ', np.sum(cm[1]))


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,50])
    plt.ylim([60,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')