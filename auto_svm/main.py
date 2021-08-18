from ae import *
from config import *
import itertools
import random
import os

import logging 

logging.basicConfig(level=logging.DEBUG, format='%(message)s', datefmt='%m-%d %H:%M', filename='tfAutoSVM.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

def get_unknown_label(train_set):
    index = random.randint(0, len(labels)-1)
    while labels[index] in train_set:
        index = random.randint(0, len(labels)-1)

    return labels[index]

def run(svms, test_datas):
    for test_set in test_datas:
        logging.info("Validate on: {}".format(test_set))
        model.validate(svms, test_set)

model = AutoSVM(logging)
model.setupModel()

if not os.path.exists('encoded_datas.pkl'):
    dataController = DataController(batch_size=batch_size, data_list=labels)
    model.train(dataController, 20)
    model.save_encoded_data()

model.load_encoded_data()
# model.test_svm(datas)
    
train_datas = list(itertools.combinations(labels, 4))
random.Random(52).shuffle(train_datas)

for train_data in train_datas:
    test_datas = []
    svms = model.train_svms(train_data)
    for i in range(8):
        unknown_label = get_unknown_label(train_data)
        test_data = list(train_data[:n_classes]) + [unknown_label]
        if test_data not in test_datas: 
            test_datas.append(test_data)
    for test_set in test_datas:
        logging.info("Validate on: {}".format(test_set))
        model.validate(test_set)



# def test_clustering(model, train_data, test_data):
#     model.sess.run(tf.global_variables_initializer())

#     trainDataController = DataController(batch_size=batch_size, data_list=train_data)
#     logging.info("Training on: {}".format(train_data))
#     model.train(trainDataController, 10)

#     testDataController = DataController(batch_size=batch_size, data_list=test_data)
#     model.load()

#     logging.info("Clustering on: {}".format(test_data))
#     label_mapping, cluster_label_counts = run_clustering(model, testDataController)
#     logging.info('--test data clsutering label_mapping {}'.format(label_mapping))
#     logging.info('--test data clsutering cluster_label_counts {}'.format(cluster_label_counts))

#     label_mapping, cluster_label_counts = run_clustering(model, trainDataController)
#     logging.info('--train data clsutering label_mapping {}'.format(label_mapping))
#     logging.info('--train data clsutering cluster_label_counts {}'.format(cluster_label_counts))

#     model.post_train(trainDataController)
#     model.load()

#     logging.info("Clustering on: {}".format(test_data))
#     label_mapping, cluster_label_counts = run_clustering(model, testDataController, validation_mode)
#     logging.info('--test data clsutering label_mapping {}'.format(label_mapping))
#     logging.info('--test data clsutering cluster_label_counts {}'.format(cluster_label_counts))

#     label_mapping, cluster_label_counts = run_clustering(model, trainDataController, validation_mode)
#     logging.info('--train data clsutering label_mapping {}'.format(label_mapping))
#     logging.info('--train data clsutering cluster_label_counts {}'.format(cluster_label_counts))


# for train_data in train_datas:
#     train_data = list(train_data)
#     test_datas = []

#     test_data = [item for item in labels if item not in train_data]
#     res = test_clustering(model, train_data, test_data)
