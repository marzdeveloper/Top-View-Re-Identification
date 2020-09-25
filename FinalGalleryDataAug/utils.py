import numpy as np
import operator
from functools import reduce
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def load_params(dir):
    return np.load(dir).item()


def flat_shape(tensor):
    """Return flattened dimension of a sample"""
    s = tensor.get_shape()
    shape = tuple([s[i].value for i in range(0, len(s))])
    return reduce(operator.mul, shape[1:])


def update_collection(name, feature, collection):
    """ Update the appropriate features from different the collection """

    # Flatten samples
    if name is not 'label':  # label arrays are already in the correct format
        batch_size = np.shape(feature)[0]
        sample_size = reduce(operator.mul, np.shape(feature)[1:])
        feature = np.reshape(feature, (batch_size, sample_size))

    # Add to the previous array
    if np.shape(collection[name])[0] == 0:  # if it's the first feature to be inserted in the collection
        collection[name] = feature
    else:
        collection[name] = np.vstack((collection[name], feature))

    return collection


def collect_features(sess, feed_dict, nodes, labels, collection):
    """ Collect the features from nodes in collection (dictionary) """

    # Generate collection dictionary to store all features if 'collection is empty
    if collection == {}:
        for node in nodes:
            collection.update({node.name: np.array([])})
        collection.update({'label': np.array([])})

    # Extract features
    features = sess.run(nodes, feed_dict=feed_dict)

    # Update collection with the extracted features
    for node, feature in zip(nodes, features):
        collection = update_collection(node.name, feature, collection)
    collection = update_collection('label', labels, collection)

    return collection


def prepare_sequence(names, features, batch_size):
    """
    Transform dictionary of features in sequential input for recurrent networks.

    names: ORDERED list of string with the name of features for sequence
    features: dict of ndarrays -- key = layer name, value = (samples x feature size)
    batch_size: desired batch size for the returned sequence
    :return:
    x: (batch_size, time_steps, input_dim)
    y: (batch_size, num_classes)
    """

    pass

    return x, y


def count_params(trainable_variables):
    global_w = 0
    for var in trainable_variables:
        shape = var.shape
        local_w = 1
        for i in range(len(shape)):
            local_w *= int(shape[i])
        global_w += local_w
    return global_w


def log_file(history_callback, log_dir, params):
    log_name = log_dir + 'log_'
    for p in params:
        log_name += ('_' + str(p))
    with open(log_name, 'w+') as f:
        num_entries = len(history_callback[log[0]])
        for i in np.arange(num_entries):
            line = log[0] + ' = ' + str(history_callback[log[0]][i]) + ' , ' + \
                   log[1] + ' = ' + str(history_callback[log[1]][i]) + ' , ' + \
                   log[2] + ' = ' + str(history_callback[log[2]][i]) + ' , ' + \
                   log[3] + ' = ' + str(history_callback[log[3]][i]) + '\n'

            f.write(line)

    print('Log file saved.\n')


# crea il log
def sLog(data, name):
    file = open(name, "a")
    file.write(str(data))
    file.close()


# allLabel:  vettore grande quanto il n di foto del test, contiene la classe originale
# allPredsProb:  vettore grande quanto il n di foto del test, contiene il vettore di probabilitÃƒÂ  delle classi
# numclasses da modificare(?)
# alllabel = fullbatch
def computeRoc(full_batch, full_pred, num_classes):
    y_true = np.array(full_batch)
    y_score = np.array(full_pred)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig3 = plt.figure(figsize=(8, 6), dpi=80)
    lw = 2

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='darkorange', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    #fig3.savefig("roc.png")
    plt.close(fig3)



#calcola la distanza euclidea tra due feature vector
#vector_1 è un vettore delle feature relativo ad un'immagine
#vector_2 è un vettore di vettori di feature, relative alla galleria
#dist è una matrice con num_classes righe e frames_gallery colonne
def euclidean_distance(vector_1, vector_2, label, num_classes, frames_gallery):
    dist = []
    for clas in range(num_classes):
        count = 0
        for i, item in enumerate(label):
            if item == clas:
                dist.append(np.linalg.norm(vector_1-vector_2[i]))
                count += 1
                if count == frames_gallery:
                    break
    return dist


#a partire dalla matrice delle distanze trova la distanza minima tra ogni frames della gallery della stessa classe e le immagini di test
#restituisce una matrice che contiene solo la distaza minima, quindi non più frames_gallery per classe nella gallery ma una per classe nella gallery
def dist_matrix_min(dist, frames_gallery):
    x = []
    for i, item in enumerate(dist):
        x.append([])
        for j in range(0, len(item), frames_gallery):
            x[i].append(min(item[j:j + frames_gallery]))
    return x

#a partire dalla matrice delle distanze calcola la distanza media tra ogni frames della gallery della stessa classe e le immagini di test
#restituisce una matrice che contiene solo le distaza medie, quindi non più frames_gallery per classe nella gallery ma una per classe nella gallery
def dist_matrix_avg(dist, frames_gallery):
    x = []
    for i, item in enumerate(dist):
        x.append([])
        for j in range(0, len(item), frames_gallery):
            x[i].append(np.mean(np.array(item[j:j + frames_gallery])))
    return x
