import os, socket
import numpy as np
import config
import json

seed = 7001

with open('classes.json') as classes_file:
    classes_config = json.load(classes_file)


# Change the path variable to the directory conataning vectorized labels
path = '/home/behzad/University/sample_data/'

if socket.gethostname() != 'HomeX':
	path = '/mnt/inl-bkp2/behzad/iscxIDS2017/pcap/'

if socket.gethostname() == 'soltani-server':
	path = '/backup3/iscxIDS2017/pcap/'

if os.getenv('DATASET') == '2018':
	path = '/mnt/INL-Backup/bkp-Home/soltani/vectorize/'

flow_size = config.flow_size
pkt_size = config.pkt_size

group_1 = ['attack_bot', 'attack_DDOS', 'DOS_GoldenEye', 'DOS_Hulk']
group_2 = ['attack_portscan']
group_3 = ['DOS_SlowHttpTest', 'DOS_SlowLoris', 'FTPPatator']
group_4 = ['Web_BruteForce', 'Web_XSS']

class DataController(object):
	def __init__(self, batch_size, data_list):
		super(DataController, self).__init__()
		self.batch_size = batch_size

		self.train_counter = 0
		self.test_counter = 0
		self.full_counter = 0

		data_dirs = []
		assert len(data_list) != 0, 'data_list argument cannot be empty'

		for i, d in enumerate(data_list):
			data_dirs.append(path + classes_config[d]['path'])
			if os.getenv('LABEL_TYPE') != 'group':
				classes_config[d]['label'] = i
			else:
				if d in group_1:
								classes_config[d]['label'] = 0
				if d in group_2:
								classes_config[d]['label'] = 1
				if d in group_3:
								classes_config[d]['label'] = 2
				if d in group_4:
								classes_config[d]['label'] = 3

		files = list()
		for directory in data_dirs:
			files.append(os.listdir(directory))

		n_per_label = 10000
		for file in files:
			n_per_label = min(n_per_label, len(file))

		data_files = list()
		for k in range(0, files.__len__()):
			for j in range(0, files[k].__len__()):
				if j > n_per_label:
					break
				data_files.append(data_dirs[k] + files[k][j])

		np.random.seed(seed)
		np.random.shuffle(data_files)
		self.data_files = data_files
		cut = int(len(data_files)*0.7)
		self.trainIDs = data_files[:cut]
		self.testIDs = data_files[cut:]

		self.trainIDs =  self.trainIDs[: len(self.trainIDs) - len(self.trainIDs)%batch_size]
		self.testIDs =  self.testIDs[: len(self.testIDs) - len(self.testIDs)%batch_size]
		
		self.train_n_batches = len(self.trainIDs) // batch_size
		self.test_n_batches = len(self.testIDs) // batch_size
		self.full_n_batches = len(self.data_files) // batch_size

		# print ('train size:' , len(self.trainIDs), 'batches:', self.train_n_batches)
		# print ('test size:' , len(self.testIDs), 'batches:', self.test_n_batches)

	def generate(self, mode='full'):
		num = 1
		if mode is 'full':
			target_set = self.data_files
			counter = self.full_counter
			n_batches = self.full_n_batches
			self.full_counter += 1
		elif mode is 'test':
			target_set = self.testIDs
			counter = self.test_counter
			n_batches = self.test_n_batches
			self.test_counter += 1
		elif mode is 'train':
			target_set = self.trainIDs
			counter = self.train_counter
			n_batches = self.train_n_batches
			self.train_counter += 1
		else:
			raise ValueError('DataGenerator mode not defined')

		start_index = counter*num*self.batch_size
		end_index = (counter+1)*num*self.batch_size

		if counter < n_batches:
			# print('data: {}/{}'.format(counter+1, n_batches))
			files = target_set[start_index : end_index]
		else:
			# print('No more data ...')
			return False

		set_x, set_y, filenames = [], [], []
		for file in files:
			X, Y = self.parse_flow(file)
			set_x.append(X)
			set_y.append(Y)
			filenames.append(file)

		set_x = np.array(set_x)
		set_y = np.array(set_y)
		filenames = np.array(filenames)
		return {
				'counter': counter,
				'x': set_x,
				'y': set_y,
				'filenames': filenames,
				'start_index': start_index,
				'end_index': end_index}
	
	def parse_flow(self, filename):
		X = list()
		Y = list()
		with open(filename) as flowfile:
			label = self.get_label_from_name(filename)
			Y.append(label)
			flowmatrix = list()
			counter = 0
			for line in flowfile:
				if counter == flow_size:
					break
				line = line[:-1].split(',')
				for i in range(len(line)):
					line[i] = float(line[i])
				if line[0] < 0:
					line[0] = 0.0
				if line[0] > 0.5:
					line[0] = 1
				else:
					line[0] = line[0] / 0.5
				for i in range(pkt_size - len(line)):#append 0 to packets which are smaller than packetsize(max=1500)
					line.append(0.0)
				line = line[:pkt_size]
				for i in range(11,21):#masking src/dst ip to 0 (12,20)+1(for timediff) ,,, maksing checksum to 0 (10,12)+1=> (11,21) -- (1,41): just payload
					line[i] = 0.0
				flowmatrix.append(line)
				counter += 1
			for i in range(flow_size - len(flowmatrix)):
				flowmatrix.append([0] * pkt_size)
			flowmatrix = np.array(flowmatrix)

		X_train = np.reshape(flowmatrix, (-1, flowmatrix.shape[0]*flowmatrix.shape[1]))
		Y_train = np.array(Y)

		if len(X_train) != 0 and len(Y_train) != 0:
			data_x, data_y = (X_train[0], Y_train[0])
		else:
			data_x, data_y = [], []
		# data_x = np.reshape(data_x, (flow_size, pkt_size))
		return data_x, data_y

	def reset(self):
		self.train_counter = 0
		self.test_counter = 0
		self.full_counter = 0

	def get_n_samples(self, mode):
		if mode is 'test':
			target_set = self.testIDs
			counter = self.test_counter
			n_batches = self.test_n_batches
			self.test_counter += 1
		else: 
			target_set = self.trainIDs
			counter = self.train_counter
			n_batches = self.train_n_batches
			self.train_counter += 1
		return len(self.trainIDs)

	def get_label_from_name(self, filename):
		for clas in classes_config:
			if filename.__contains__(clas):
				return classes_config[clas]['label']
			
			if os.getenv('DATASET') == '2018':
				if '2018' in clas:
					#print(filename, clas.split('-'))
					if filename.__contains__(clas.split('-')[1]):
						return classes_config[clas]['label']
