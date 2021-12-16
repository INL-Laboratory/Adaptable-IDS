import tensorflow as tf
import socket
if socket.gethostname() == 'HomeX':
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.utils import shuffle
import pickle
import random
from clustering import *

import config
from data import DataController

learning_rate = 0.01
num_steps = 50

display_step = 1000
examples_to_show = 10

num_hidden_1 = 5000
num_hidden_2 = 2000
num_hidden_3 = 1000
batch_size = 100

flow_size = config.flow_size
pkt_size = config.pkt_size
num_input = flow_size * pkt_size
labels = config.all_labels
n_classes = config.n_classes


weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([num_hidden_1, num_input])),

}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([num_input])),
}

class AutoSVM(object):
    def __init__(self, logging):
        self.input_size = num_input
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2

        self.X = tf.placeholder("float", [None, self.input_size])
        
        self.batch_size = batch_size
        self.sess = tf.InteractiveSession()
        self.logger = logging
        self.saver = tf.train.Saver()

    def encoder(self, x):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                    biases['encoder_b2']))

        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                                biases['encoder_b3']))
        self.output = layer_3
        return layer_3

    def decoder(self, x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                    biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                    biases['decoder_b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                                biases['decoder_b3']))
        return layer_3
    
    def setupModel(self):
        self.encoder_op = self.encoder(self.X)
        self.decoder_op = self.decoder(self.encoder_op)
    
    def _same_centroid_loss(self, last_layer_outputs, centroids):
        loss = 0
        for i in range(self.batch_size):
            loss += tf.linalg.norm(last_layer_outputs[i] - centroids[i])
        return tf.reduce_sum(loss)

    def post_train(self, dataController, epochs=10):
        self.nearest_centroids = tf.placeholder(tf.float32, shape=[None, num_hidden_3], name='C')

        post_loss = self._same_centroid_loss(self.output, self.nearest_centroids)
        post_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(post_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        for i in range(1, epochs+1):
            epoch_losses = []
            print('Epoch => {}'.format(i))
            dataController.reset()
            while 1:
                data = dataController.generate('train')
                if data is False:
                    break
                x = data["x"]
                y = data["y"]
                names = data["filenames"]
                centers = get_nearest_same_label_centeroid(y)
                #print(centers[0])

                feed_dict_batch = {self.X: x, 
                                   self.nearest_centroids: centers}
                _, batch_loss = self.sess.run((post_optimizer, post_loss), feed_dict=feed_dict_batch)
                # print(batch_loss)
                epoch_losses.append(batch_loss)
                # if i % display_step == 0 or i == 1:
            print('Post Train Step %i: Loss: %f' % (i, batch_loss))
            self.save('autosvm_post_train_model')

    def train(self, dataController, epochs=20):
        y_pred = self.decoder_op
        y_true = self.X

        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        dataController = DataController(batch_size=batch_size, data_list=labels)
        for i in range(1, epochs+1):
            dataController.reset()
            while 1:
                # print(i)
                data = dataController.generate('train')
                if data is False:
                    break
                x = data["x"]
                _, l = self.sess.run([optimizer, loss], feed_dict={self.X: x})
                # if i % display_step == 0 or i == 1:
            self.save()
            print('Step %i: Minibatch Loss: %f' % (i, l))
    
    def encode(self, x):
        encoded = self.sess.run(self.encoder_op, feed_dict={self.X: x})
        return encoded
    
    def save(self, path='./saved_autosvm/'):
        self.saver.save(self.sess, path)
    
    def load(self, path='./saved_autosvm/'):
        self.saver.restore(self.sess, path)

    def save_encoded_data(self, file_name='encoded_datas'):
        datas = {}
        for label in labels:
            print("Getting encoded data for", label)
            datas[label] = []
            dataController = DataController(batch_size=batch_size, data_list=[label])
            while 1:
                data = dataController.generate('test')
                if data is False:
                    break
                x = data["x"]
                enc_out = self.encode(x)
                datas[label].extend(enc_out)
            datas[label] = np.array(datas[label])

        with open(file_name + '.pkl', 'wb') as f:
            pickle.dump(datas, f, pickle.HIGHEST_PROTOCOL)
    
    def load_encoded_data(self, file_name='encoded_datas'):
        with open(file_name + '.pkl', 'rb') as f:
            self.datas = pickle.load(f)
        return self.datas


    def test_svm(self, datas):
        for k, v in datas.items():
            print(k, len(v))
        # exit()
        kernels = ['rbf']
        nus = [0.001]
        svms = []
        for label in labels:
            for kernel in kernels:
                for nu in nus:
                    self.logger.info('=========================')
                    self.logger.info('Test on {} {} {}'.format(label, kernel, nu))
                    X_0 = list()
                    train_labels = random.sample(labels, 3)
                    print('random labels', train_labels)
                    for t_label in train_labels:
                        X_0.extend(datas[t_label])

                    X_1 = datas[label]

                    X1_train = X_1[0:int(len(X_1)*0.7)]
                    X1_test = X_1[int(len(X_1)*0.7):-1]

                    X0_train = X_0[0:int(len(X_0)*0.7)]
                    X0_test = X_0[int(len(X_0)*0.7):-1]


                    Y1_train = [1] * len(X1_train)
                    Y1_test = [1] * len(X1_test)
                    Y0_train = [0] * len(X0_train)
                    Y0_test = [0] * len(X0_test)


                    X_train =  np.concatenate((X1_train, X0_train))
                    Y_train =  np.concatenate((Y1_train, Y0_train))

                    X_test =  np.concatenate((X1_test, X0_test))
                    Y_test =  np.concatenate((Y1_test, Y0_test))

                    clf = svm.SVC(kernel=kernel, probability=True)

                    clf.fit(X_train, Y_train)

                    # y1_pred = clf.predict(X1_test)
                    # y1_scores = clf.predict_proba(X1_test)
                    # print(y1_scores)
                    # print("Accuracy1:",metrics.accuracy_score(Y1_test, y1_pred))

                    # y0_pred = clf.predict(X0_test)
                    # y0_scores = clf.predict_proba(X0_test)
                    # print(y0_scores)
                    # print("Accuracy0:",metrics.accuracy_score(Y0_test, y0_pred))

                    # ocsvm = svm.OneClassSVM(nu=nu, kernel=kernel)
                    # samples = self.datas[label]
                    # train_samples = samples[0:int(len(samples)*0.7)]
                    # test_samples = samples[int(len(samples)*0.7):-1]

                    # ocsvm.fit(train_samples)
                    svms.append(clf)
            # continue
            for ocsvm in svms:
                # correct = 0
                # for i, sample in enumerate(test_samples):
                #     scores = ocsvm.predict_proba([sample])
                #     predict = ocsvm.predict([sample])[0]
                #     if predict == 1:
                #         correct += 1

                # self.logger.info('Known Acc: {} {}'.format(label, str(round(100*(correct/len(test_samples)), 2))))

                for other_label in labels:
                    if other_label == label or other_label in train_labels:
                        continue

                    samples = self.datas[other_label]
                    truth = [0] * len(samples)
                    y1_pred = ocsvm.predict(samples)
                    acc = metrics.accuracy_score(truth, y1_pred)
                    scores = ocsvm.predict_proba(samples)
                    print(scores)

                    # for i, sample in enumerate(samples):
                    #     score = ocsvm.predict_proba([sample])
                    #     print(score)
                    #     predict = ocsvm.predict([sample])[0]
                    #     if predict == -1:
                    #         correct += 1

                    self.logger.info('Unknown Acc: {}, {}'.format(other_label, acc))


    def _get_binary_train_data(self, label_1, labels):
        X_0 = list()
        labels.remove(label_1)
        for label_0 in labels:
            X_0.extend(self.datas[label_0])
        random.shuffle(X_0)


        X_1 = self.datas[label_1]
        train_size_1 = int(len(X_1)*0.7)

        X1_train = X_1[0:train_size_1]
        X1_test = X_1[train_size_1:-1]

        train_size_0 = int(len(X_0)*0.7)
        X0_train = X_0[0:train_size_0]
        X0_test = X_0[train_size_0:-1]


        Y1_train = [1] * len(X1_train)
        Y1_test = [1] * len(X1_test)
        Y0_train = [0] * len(X0_train)
        Y0_test = [0] * len(X0_test)



        X_train =  np.concatenate((X1_train, X0_train))
        Y_train =  np.concatenate((Y1_train, Y0_train))

        X_train, Y_train = shuffle(X_train, Y_train, random_state=0)


        X_test =  np.concatenate((X1_test, X0_test))
        Y_test =  np.concatenate((Y1_test, Y0_test))

        X_test, Y_test = shuffle(X_test, Y_test, random_state=0)


        return X_train, Y_train, X_test, Y_test

    def train_svms(self, train_labels):
        self.svms = []
        self.logger.info("Training on: {}".format(train_labels))
        for i, train_label in enumerate(train_labels):
            train_size = int(len(self.datas[train_label])*0.7)
            clf = svm.SVC(kernel='rbf', probability=True)
            X_train, Y_train, _, _ = self._get_binary_train_data(train_label, list(train_labels))
            clf.fit(X_train, Y_train)
            self.svms.append(clf)

        return self.svms

        # self.svms = []
        # self.logger.info("Training on: {}".format(train_labels))
        # for i, train_label in enumerate(train_labels):
        #     train_size = int(len(self.datas[train_label])*0.7)
        #     ocsvm = svm.OneClassSVM(nu=0.001, kernel='rbf', gamma='auto')
        #     ocsvm.fit(self.datas[train_label][0:train_size])
        #     self.svms.append(ocsvm)

        # return self.svms


    def validate(self, labels):
        label2int = {}
        res = {}
        for i, label in enumerate(labels):
            label2int[label] = i
            res[i] = [0, 0, 0, 0]

        # print(label2int)
        # print(res)

        samples = []
        truths = []
        for label in labels:
            train_size = int(len(self.datas[label])*0.7)
            samples.extend(self.datas[label][train_size:-1])
            label_int = label2int[label]
            truths.extend([label_int]*len(self.datas[label][train_size:-1]))
        print(len(samples), len(truths))

        unknown_count = 0
        scores = []
        predicts = []
        for k, clf in enumerate(self.svms):
            score = clf.predict_proba(samples)
            scores.append(score)

            predict = clf.predict(samples)
            predicts.append(predict)

        for i, sample in enumerate(samples):
            pred = -1
            score = 0
            for k in range(len(self.svms)):
                if predicts[k][i] == 1:
                    if scores[k][i][1] > 0.98 and scores[k][i][1] > score:
                        score = scores[k][i][1]
                        pred = k
            # print('top score', score)
            if pred == -1:
                pred = n_classes


            res[truths[i]][0] += 1

            if pred == truths[i]:
                res[pred][1] += 1
            
            if truths[i] == n_classes and pred != n_classes:
                res[pred][3] += 1

            res[truths[i]][2] = res[truths[i]][1]/res[truths[i]][0] * 100

        # for i, sample in enumerate(samples):
        #     res[truths[i]][0] += 1
        #     predicts = []
        #     scores = []
        #     for k, clf in enumerate(self.svms):
        #         score = clf.predict_proba([sample])[0]
        #         scores.append(score)

        #         predict = ocsvm.predict([sample])[0]
        #         if predict == 1:
        #             predicts.append(k)

        #     # predict = np.argmax(scores)
        #     # if scores[predict] > 0.09:
        #     #     pred = predict
        #     #     # predicts.append(predict)
        #     # else:
        #     #     unknown_count += 1
        #     #     pred = n_classes

        #     pred = None
        #     if len(predicts) == 0:
        #         unknown_count += 1
        #         pred = n_classes
        #     else:
        #         pred = predicts[0]

        #     if pred == truths[i]:
        #         res[pred][1] += 1

        #     if truths[i] == n_classes and pred != n_classes:
        #         res[pred][3] += 1

        #     res[truths[i]][2] = res[truths[i]][1]/res[truths[i]][0] * 100
        #     # print(truths[i])
        #     # print(scores)
        #     # print(predicts)
        self.logger.info(res)

    def find_threshold(self, dataController):
        dataController.reset()
        res = {}
        assignments = []
        centroids = []

        max_same = 0
        min_diff = 1000000

        while 1:
            data = dataController.generate()
            if data is False:
                break
            x = data["x"]
            y = data["y"]
            names = data["filenames"]
            feed_dict_batch = {self.X: x}
            
            output = self.encode(x)

            # print(output)
            
            for p, i in enumerate(output):
                for q, j in enumerate(output):
                    if y[p] == y[q]:
                        max_same = distance(i, j) if distance(i, j) > max_same else max_same
                    if y[p] != y[q]:
                        min_diff = distance(i, j) if distance(i, j) < min_diff else min_diff

        print(min_diff, max_same)

        return min_diff, max_same
        
            # assigns = cluster(output)


        return res
    def create_mapping(self, dataController):
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
            feed_dict_batch = {self.X: x}
            
            output = self.encode(x)

            # print(output)
            for p, i in enumerate(output):
                for q, j in enumerate(output):
                    print(y[p], y[q], distance(i, j))
            assigns = cluster(output)

            for i, name in enumerate(names):
                res[name] = assigns[i]

        return res
