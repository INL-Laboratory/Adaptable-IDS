import os

flow_size = 100
pkt_size = 200
n_classes = 4
learning_rate = 0.0001
batch_size = 20
epochs = 20
doc_uknown_threshold = 0.8
openmax_uknown_threshold = 0.9

all_labels = ['vectorize_friday/benign', 'attack_bot', 'attack_DDOS',\
            'attack_portscan', 'Benign_Wednesday', 'DOS_SlowHttpTest',\
            'DOS_SlowLoris', 'DOS_Hulk', 'DOS_GoldenEye', 'FTPPatator',\
            'SSHPatator', 'Web_BruteForce', 'Web_XSS']

if os.getenv('DATASET') == '2018':
  all_labels = ['2018-Benign', '2018-Bot', '2018-Web', '2018-SQL', \
              '2018-Infilteration', '2018-GoldenEye', '2018-Slowloris',\
              '2018-DDoS', '2018-FTP', '2018-SSH']