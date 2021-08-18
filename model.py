import socket
import tensorflow as tf
if socket.gethostname() == 'HomeX':
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt

import scipy.spatial.distance as spd
from openmax_utils.compute_openmax import recalibrate_scores
from openmax_utils.evt_fitting import weibull_tailfitting
from sklearn.metrics import accuracy_score

from clustering import *


import config
flow_size = config.flow_size
pkt_size = config.pkt_size
learning_rate = config.learning_rate
epochs = config.epochs
doc_uknown_threshold = config.doc_uknown_threshold
openmax_uknown_threshold = config.openmax_uknown_threshold
n_classes = config.n_classes
label = [i for i in range(n_classes)]

#------------------------OpenMax-------
def compute_distances(mean_feature, feature):
    eucos_dist, eu_dist, cos_dist = [], [], []
    eu_dist, cos_dist, eucos_dist = [], [], []
    for feat in feature:
        eu_dist += [spd.euclidean(mean_feature, feat)]
        cos_dist += [spd.cosine(mean_feature, feat)]
        eucos_dist += [spd.euclidean(mean_feature, feat)/200. + spd.cosine(
            mean_feature, feat)]
    distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}
    return distances

def build_weibull(mean, distance, tail):
    weibull_model = {}
    for i in range(len(mean)):
        weibull_model[label[i]] = {}
        weibull = weibull_tailfitting(mean[i], distance[i], tailsize=tail)
        weibull_model[label[i]] = weibull
    return weibull_model
#-------------------------------------


class Model(object):
    def __init__(self, input_size, n_classes, batch_size, loss_function, logging):
        self.input_size = input_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.sess = tf.InteractiveSession()
        self.logger = logging
        
    
    def build_base(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_size], name='X')
        self.y = tf.placeholder(tf.int32, shape=[None], name='Y')
        self.nearest_centroids = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='C')
        self.output = None
        self.input = tf.reshape(self.x, shape=[-1, flow_size, pkt_size])

    def build_dense(self):
        shape = self.output.get_shape().as_list()
        dim = np.prod(shape[1:])
        self.output = tf.reshape(self.output, [-1, dim])

        # self.test = self.output
        self.output = tf.layers.dense(self.output, 500, activation=tf.nn.relu6)
        
        # self.output = tf.layers.dense(self.output, 100, activation=tf.nn.sigmoid)
        self.output = tf.layers.dense(self.output, self.n_classes)
        self.activationVector = self.output
        self.output = tf.nn.sigmoid(self.output)

        self.saver = tf.train.Saver()

    def build_lstm_model(self):
        
        self.build_base()

        #--------Build LSTM------
        n_hidden = 512
        rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, dtype=tf.float32)
        self.output, states = tf.nn.dynamic_rnn(rnn_cell, self.input, dtype=tf.float32)
        #-----------------------

        # self.test = self.output
        self.build_dense()
        
        return self.output

    def build_model(self):
        
        self.build_base()

        #--------Build CNN------
        output_channels = 10
        filter_size = 20
        w = tf.Variable(tf.random_normal([filter_size, pkt_size, output_channels]))
        # b = tf.Variable(tf.random_normal([output_channels]))

        self.output = tf.nn.conv1d(self.input, w, stride=1, padding="VALID")
        self.output = tf.layers.max_pooling1d(self.output, pool_size=2, strides=1, padding='VALID')
        #-----------------------

        self.build_dense()
        
        return self.output

    def build_classification(self):
        self.preds = tf.argmax(self.output, axis=1, name='predictions')
        if self.loss_function == '1-vs-rest':
            self.loss = self.one_vs_rest_loss(self.output, self.y)
        elif self.loss_function == 'softmax':
            self.output = tf.nn.softmax(self.activationVector)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y, self.n_classes), logits=self.activationVector))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(self.loss)
        self.post_loss = self.same_centroid_loss(self.output, self.nearest_centroids)
        self.post_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(self.post_loss)

    def one_vs_rest_loss(self, output, labels):
        loss = 0
        for i in range(self.n_classes):
            for j in range(self.batch_size):
                loss -= tf.cond(tf.equal(labels[j], i), lambda: tf.log(output[j][i]), lambda: tf.log(1 - output[j][i]))
        # return loss -= tf.cond(tf.equal(labels[j], i), lambda: tf.log(output[j][i]), lambda: tf.log(1 - output[j][i]))
        return tf.reduce_sum(loss)

    def same_centroid_loss(self, last_layer_outputs, centroids):
        loss = 0
        for i in range(self.batch_size):
            loss += tf.linalg.norm(last_layer_outputs[i] - centroids[i])
        return tf.reduce_sum(loss)

    def load(self, path='./saved_model/'):
        self.saver.restore(self.sess, path)

    def save(self, path='./saved_model/'):
        self.saver.save(self.sess, path)

    def train(self, dataController):
        self.logger.info('---Starting train with {} epochs'.format(epochs))
        # print('Loss function: DOC Sigmoid')
        losses = []
        for epoch in range(epochs):
            epoch_losses = []
            # print('Epoch => {}'.format(epoch))
            dataController.reset()
            while 1:
                data = dataController.generate('train')
                if data is False:
                    break
                counter = data["counter"]
                x = data["x"]
                y = data["y"]
                names = data["filenames"]
                feed_dict_batch = {self.x: x, 
                                   self.y: y}
                _, batch_loss, output= self.sess.run((self.optimizer, self.loss, self.output), feed_dict=feed_dict_batch)
                # if counter % 20 == 0:
                    # print(test.shape)
                    # print(counter)
                    # print(y)
                    # print(batch_loss)
                    # print(output)
                    # print(test)
                epoch_losses.append(batch_loss)
            epoch_loss_mean = np.mean(epoch_losses)
            losses.append(epoch_loss_mean)
            if epoch % 10 == 0:
                self.save()
                # print("Model Saved!")
                self.logger.info("Loss={}".format(epoch_loss_mean))
                plt.plot(losses)
                plt.savefig('loss.png')
                # plt.show()

    def post_train(self, dataController):
        self.logger.info('---Post Train with {} epochs'.format(epochs))
        print('Loss function: Distance to same labeled centeroid')
        for epoch in range(epochs):
            epoch_losses = []
            print('Epoch => {}'.format(epoch))
            dataController.reset()
            while 1:
                data = dataController.generate('train')
                if data is False:
                    break
                x = data["x"]
                y = data["y"]
                names = data["filenames"]
                centers = get_nearest_same_label_centeroid(y)

                feed_dict_batch = {self.x: x, 
                                   self.y: y,
                                   self.nearest_centroids: centers}
                _, batch_loss, output = self.sess.run((self.post_optimizer, self.post_loss, self.output), feed_dict=feed_dict_batch)
                # print(batch_loss)
                # print(output)
                # print(test)
                epoch_losses.append(batch_loss)

            if epoch % 1 == 0:
                self.save("./post_train_phase/")
                print("Model Saved!")
                print("Loss={}".format(np.mean(epoch_losses)))

    def predict_openmax(self, data):
        alpharank_list = [4]
        # tail_list = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        tail_list = [20]
        # total = 0
        for alpha in alpharank_list:
            weibull_model = {}
            openmax = None
            softmax = None
            for tail in tail_list:
                # print ('Alpha ',alpha,' Tail ',tail)
                # print ('++++++++++++++++++++++++++++')
                weibull_model = build_weibull(self.feature_mean, self.feature_distance, tail)
                openmax, softmax = recalibrate_scores(
                    weibull_model, label, data, alpharank=alpha)
        # if openmax == 4:
        #     openmax = 'Unknown'
        # print('Prediction openmax: ', openmax)
        return softmax, openmax
        
    def calc_mean_and_dist(self, dataController, unknown_label):
        dataController.reset()
        self.feature_mean = [[] for i in range(self.n_classes)]
        self.feature_distance = [[] for i in range(self.n_classes)]
        while 1:
            data = dataController.generate('train')
            if data is False:
                break
            x = data["x"]
            y = data["y"]
            names = data["filenames"]
            feed_dict_batch = {self.x: x,
                               self.y: y,
                              }
            output, activationVector = self.sess.run((self.output, self.activationVector), feed_dict=feed_dict_batch)
            
            for i in range(self.batch_size):
                if y[i] == unknown_label:
                    continue
                if len(self.feature_mean[y[i]]) == 0:
                    self.feature_mean[y[i]] = activationVector[i]
                else:
                    self.feature_mean[y[i]] = (self.feature_mean[y[i]] + activationVector[i]) / 2
            
            for i in range(self.batch_size):
                if y[i] == unknown_label:
                    continue
                if len(self.feature_distance[y[i]]) == 0:
                    self.feature_distance[y[i]] = compute_distances(self.feature_mean[y[i]], activationVector[i])
                else:
                    distance = compute_distances(self.feature_mean[y[i]], activationVector[i])
                    for key in distance.keys():
                        self.feature_distance[y[i]][key] += distance[key]

    def validate(self, dataController, unknown_label, mode='DOC'):
        # print('---Runnning validate function---')
        dataController.reset()
        all_acc = []
        res_before_threshold = {}
        res = {}
        for i in range(self.n_classes+1):
            res[i] = [0,0,0,0]
            res_before_threshold[i] = [0,0,0,0]
        while 1:
            data = dataController.generate('test')
            if data is False:
                break
            x = data["x"]
            y = data["y"]
            names = data["filenames"]
            feed_dict_batch = {self.x: x,
                               self.y: y,
                              }
            output, activationVector = self.sess.run((self.output, self.activationVector), feed_dict=feed_dict_batch)
            
            if mode == 'DOC':
                max_indices = np.argmax(output, axis=1)
                for i in range(self.batch_size):
                    # if y[i] == unknown_label: print('-', y[i], max_indices[i], output[i][max_indices[i]])
                    # else: print(y[i], max_indices[i], output[i][max_indices[i]])

                    # self.logger.info("Befor Threshold --> Truth: {}    DOC: {}".format(y[i], max_indices[i]))
                    res_before_threshold[y[i]][0] += 1
                    #print(y, max_indices)
                    if y[i] == max_indices[i]:
                        res_before_threshold[y[i]][1] += 1
                    elif y[i] == unknown_label:
                        res_before_threshold[max_indices[i]][3] += 1
                    if res_before_threshold[y[i]][0] != 0:
                        res_before_threshold[y[i]][2] = res_before_threshold[y[i]][1]/res_before_threshold[y[i]][0] * 100

                    if output[i][max_indices[i]] < doc_uknown_threshold:
                        max_indices[i] = unknown_label
                    doc_predict = max_indices[i]
                    # self.logger.info("After Threshold --> Truth: {}    DOC: {}".format(y[i], doc_predict))

                    res[y[i]][0] += 1
                    if y[i] == max_indices[i]:
                        res[y[i]][1] += 1
                    elif y[i] == unknown_label:
                        res[max_indices[i]][3] += 1

                    if res[y[i]][0] != 0:
                        res[y[i]][2] = res[y[i]][1]/res[y[i]][0] * 100

            elif mode == 'OpenMax':
                max_indices = np.argmax(output, axis=1)
                for i in range(self.batch_size):
                    data = {}
                    data['fc8'] = np.array([activationVector[i]], dtype=np.float32)
                    data['scores'] = np.array([output[i]], dtype=np.float32)
                    softmax_scores, openmax_scores = self.predict_openmax(data)
                    openmax_predict = np.argmax(openmax_scores)
                    softmax_predict = np.argmax(softmax_scores)

                    # self.logger.info("Befor Threshold --> Truth: {}    OpenMax: {}     Softmax: {}".format(\
                    #                 y[i], openmax_predict, softmax_predict))
                    # if y[i] == unknown_label and 
                    res_before_threshold[y[i]][0] += 1
                    if y[i] == openmax_predict:
                        res_before_threshold[y[i]][1] += 1
                    elif y[i] == unknown_label:
                        res_before_threshold[max_indices[i]][3] += 1

                    if res_before_threshold[y[i]][0] != 0:
                        res_before_threshold[y[i]][2] = res_before_threshold[y[i]][1]/res_before_threshold[y[i]][0] * 100

                    if openmax_predict == unknown_label or \
                       openmax_scores[openmax_predict] < openmax_uknown_threshold:
                        max_indices[i] = unknown_label
                    # self.logger.info('----------------')
                    # self.logger.info('openmax_scores: {}'.format(openmax_scores))
                    # self.logger.info('softmax_scores: {}'.format(softmax_scores))
                    # self.logger.info('')
                    # self.logger.info("After Threshold --> Truth: {}    OpenMax: {}".format(\
                                    # y[i], max_indices[i]))
                    res[y[i]][0] += 1
                    if y[i] == max_indices[i]:
                        res[y[i]][1] += 1
                    elif y[i] == unknown_label:
                        res[max_indices[i]][3] += 1

                    if res[y[i]][0] != 0:
                        res[y[i]][2] = res[y[i]][1]/res[y[i]][0] * 100

            batch_acc = (accuracy_score(y, max_indices, normalize=False)/self.batch_size)*100
            all_acc.append(batch_acc)

        self.logger.info('Accuracy of classification {}%'.format(np.mean(all_acc)))
        self.logger.info(res_before_threshold)
        self.logger.info(res)
        final_res = [res_before_threshold, res]
        return res
        
    def create_mapping(self, dataController, validation_mode):
        #Create dict to map file names to cluster ids
        print("---Clustering on trained data output.")
        dataController.reset()
        res = {}
        assignments = []
        centroids = []
        while 1:
            data = dataController.generate('train')
            if data is False:
                break
            x = data["x"]
            y = data["y"]
            names = data["filenames"]
            feed_dict_batch = {self.x: x, 
                               self.y: y}
            output, activationVector = self.sess.run((self.output, self.activationVector), feed_dict=feed_dict_batch)
            # output = self.sess.run((self.output), feed_dict=feed_dict_batch)
            if validation_mode == 'OpenMax':
                new_output = []
                for i in range(self.batch_size):
                    data = {}
                    data['fc8'] = np.array([activationVector[i]], dtype=np.float32)
                    data['scores'] = np.array([output[i]], dtype=np.float32)
                    _, openmax_scores = self.predict_openmax(data)
                    new_output.append(openmax_scores[0:n_classes])
                output = new_output

            # print(output)
            # for p, i in enumerate(output):
            #     for q, j in enumerate(output):
            #         print(y[p], y[q], distance(i, j))
            assigns = cluster(output)

            for i, name in enumerate(names):
                res[name] = assigns[i]

        return res
