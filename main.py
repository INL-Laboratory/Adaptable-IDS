import socket
import tensorflow as tf
if socket.gethostname() == 'HomeX':
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
import config
from data import DataController

from model import Model

import logging
import itertools
import random

from clustering import *

logging.basicConfig(level=logging.DEBUG, format='%(message)s', datefmt='%m-%d %H:%M', filename='main.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--validation_mode', action='store', type=str, default='DOC++')
parser.add_argument('--arch', action='store', type=str, default='CNN')
parser.add_argument('--loss_function', action='store', type=str, default='1-vs-rest')
parser.add_argument('--target', action='store', type=str, default='classification')

args = parser.parse_args()

validation_mode = args.validation_mode
target = args.target
arch = args.arch
loss_function = args.loss_function

logging.info('---- Running main with arch {}, validation mode {}'.format(arch, validation_mode))

batch_size = config.batch_size
flow_size = config.flow_size
pkt_size = config.pkt_size
input_size = flow_size * pkt_size
n_classes = config.n_classes
all_labels = config.all_labels

def run(model, train_set=[], test_sets=[]):
    res = {}
    train_key = ' '.join([str(elem) for elem in train_set])
    res['train_key'] = ' '.join([str(elem) for elem in train_set])
    dataController = DataController(batch_size=batch_size, data_list=train_set)
    logging.info("Training on: {}".format(train_set))
    model.train(dataController)
    for test_set in test_sets:
        test_key = ' '.join([str(elem) for elem in test_set])
        res[test_key] = None

        dataController = DataController(batch_size=batch_size, data_list=test_set)
        model.load()
        if validation_mode == 'OpenMax':
            model.calc_mean_and_dist(dataController, 4)
        logging.info("Validate on: {}".format(test_set))
        out = model.validate(dataController, 4, mode=validation_mode)
        res[test_key] = out
    
    return res

def get_unknown_label(train_set):
    index = random.randint(0, len(all_labels)-1)
    while all_labels[index] in train_set:
        index = random.randint(0, len(all_labels)-1)

    return all_labels[index]


def test_clustering(model, train_data, test_data):
    trainDataController = DataController(batch_size=batch_size, data_list=train_data)
    logging.info("Training on: {}".format(train_data))
    model.train(trainDataController)

    testDataController = DataController(batch_size=batch_size, data_list=test_data)
    model.load()
    if validation_mode == 'OpenMax':
            model.calc_mean_and_dist(trainDataController, 4)

    logging.info("Clustering on: {}".format(test_data))
    label_mapping, cluster_label_counts = run_clustering(model, testDataController, validation_mode)
    logging.info('--test data clsutering label_mapping {}'.format(label_mapping))
    logging.info('--test data clsutering cluster_label_counts {}'.format(cluster_label_counts))

    label_mapping, cluster_label_counts = run_clustering(model, trainDataController, validation_mode)
    logging.info('--train data clsutering label_mapping {}'.format(label_mapping))
    logging.info('--train data clsutering cluster_label_counts {}'.format(cluster_label_counts))

    model.post_train(trainDataController)
    model.load()

    logging.info("Clustering on: {}".format(test_data))
    label_mapping, cluster_label_counts = run_clustering(model, testDataController, validation_mode)
    logging.info('--test data clsutering label_mapping {}'.format(label_mapping))
    logging.info('--test data clsutering cluster_label_counts {}'.format(cluster_label_counts))

    label_mapping, cluster_label_counts = run_clustering(model, trainDataController, validation_mode)
    logging.info('--train data clsutering label_mapping {}'.format(label_mapping))
    logging.info('--train data clsutering cluster_label_counts {}'.format(cluster_label_counts))


model = Model(input_size, n_classes, batch_size, loss_function, logging)
if arch == 'LSTM':
    model.build_lstm_model()
elif arch == 'CNN':
    model.build_model()
model.build_classification()
model.sess.run(tf.global_variables_initializer())

traning_size = n_classes

train_datas = list(itertools.combinations(all_labels, traning_size))
random.Random(52).shuffle(train_datas)

if target == 'classification':
    for train_data in train_datas:
        train_data = list(train_data)
        test_datas = []
        if validation_mode == 'DOC++':
            random_train_label = get_unknown_label(train_data)
            train_data.append(random_train_label)
        for i in range(8):
            unknown_label = get_unknown_label(train_data)
            test_data = list(train_data[:traning_size]) + [unknown_label]
            if test_data not in test_datas:
                test_datas.append(test_data)
        res = run(model, train_data, test_datas)
        # with open(res['train_key'], 'w') as f:
        #     print(res, file=f)

        model.sess.run(tf.global_variables_initializer())
else:
    for train_data in train_datas:
        train_data = list(train_data)
        test_datas = []
        if validation_mode == 'DOC++':
            random_train_label = get_unknown_label(train_data)
            train_data.append(random_train_label)
        
        test_data = [item for item in all_labels if item not in train_data]

        res = test_clustering(model, train_data, test_data)
        model.sess.run(tf.global_variables_initializer())
