import ast
import copy
import json
import copy

import csv
def generate_csv(name, results):
    csv_columns = results[0].keys()
    csv_file = name+".csv"
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in results:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def extract_clusters(name):
  f = open(name, "r")
  content = f.read()

  content_list = content.splitlines()
  f.close()

  expriments = []
  midlines = []

  def parse_res(line):
    a = line.index('[')
    b = line.index(']')
    clusters = ast.literal_eval(line[a:b+1])
    new_clusters = []
    label_counts = {}
    for cluster in clusters:
      for label in cluster.keys():
        if label in label_counts:
          label_counts[label] += cluster[label]
        else:
          label_counts[label] = cluster[label]
    for cluster in clusters:
      # print(cluster)
      new_cluster = {}
      total = 0
      for label in cluster.keys():
        total += cluster[label]
      
      for label in cluster.keys():
        if cluster[label] > total * 0.1 and total > 100:
          new_cluster[label] = str(round(100 * (cluster[label]/total), 2)) + '|' + str(round(100 * (cluster[label]/label_counts[label]), 2))
      if len(new_cluster.keys()) > 0:
        new_clusters.append(new_cluster)
      # cluster = new_cluster
      # print(cluster)
    new_clusters = ' '.join([json.dumps(i) for i in new_clusters])    
    
    return new_clusters

  def extract_data(expriment_lines):
    data = []
    post_train = False
    train_set = None
    test_set = None
    train_res = None
    test_res = None
    train_res_p = None
    test_res_p = None

    for i, line in enumerate(expriment_lines):
      if 'Training on' in line:
        a = line.index('[')
        b = line.index(']')
        # data.append('T' + line[a+1:b])
        train_set = 'T' + line[a+1:b]
      
      if 'Post Train' in line:
        post_train = True

      if ('Clustering on' in line or 
          'Unknown Test data' in line):
        a = line.index('[')
        b = line.index(']')
        # data.append(line[a+1:b])
        test_set = line[a+1:b]
      
      if '--train data clsutering cluster_label_counts' in line:
        if post_train:
          train_res_p = parse_res(line)
        else:
          train_res = parse_res(line)

        # data.append(new_clusters)
      if '--test data clsutering cluster_label_counts' in line:
        if post_train:
          test_res_p = parse_res(line)
        else:
          test_res = parse_res(line)

    data.append(train_set)
    data.append(train_res)
    data.append(train_res_p)

    data.append(test_set)
    data.append(test_res)
    data.append(test_res_p)


    return data
    

  for i, line in enumerate(content_list):
      if midlines and ('Training on' in line or i == len(content_list)):
          expriments.append(midlines)
          midlines = []
      midlines.append(line)

  expriments = expriments[1:]

  print(expriments[0])
  print('-------------------')
  print(expriments[-1])

  data = extract_data(expriments[0])

  datas = []
  for expriment in expriments:
      data = extract_data(expriment)
      datas.extend(data)


  # for i in datas:
  #   print(i)


  with open(name + '.list', 'w') as f:
      for item in datas:
          f.write("%s\n" % item)


def find_similiars(name):
  res = {}
  res['train'] = {}
  res['test'] = {}

  f = open(name, "r")
  content = f.read()
  def find_char_for_label(all_labels, label):
    res = all_labels.index(label)
    if res > 9:
      res = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][res%10]
    return str(res)


  content_list = content.splitlines()
  f.close()
  all_labels = ["'Web_XSS'", "'DOS_Hulk'", "'attack_portscan'", "'DOS_GoldenEye'", \
   "'DOS_SlowLoris'", "'FTPPatator'", "'attack_DDOS'", "'DOS_SlowHttpTest'",\
   "'attack_bot'", "'Web_BruteForce'", "'SSHPatator'"]

  for i, line in enumerate(content_list):
    if not line:
      continue
    train = False

    if line[0] == 'T' or line[0] == "'":
      if line[0] == 'T':
        labels = line[1:].split(', ')
        # if i > 0:
        #   break
      else:
        labels = line.split(', ')
      
      clusters_line = content_list[i+2]
      clusters = []
      for t, item in enumerate(clusters_line.split(' {')):
        if t != 0:
          item = '{' + item
        
        clusters.append(ast.literal_eval(item))
      
      # print(all_labels)
      # print(labels)
      # print(clusters)
      # print()

      for cluster in clusters:
        keys = list(cluster.keys())
        for j in keys:
          for k in keys:
            # if j == k:
            #   continue
            if line[0] == 'T':
              if not labels[int(j)] in res['train'].keys():
                res['train'][labels[int(j)]] = find_char_for_label(all_labels, labels[int(k)])
              else:
                res['train'][labels[int(j)]] += find_char_for_label(all_labels, labels[int(k)])
            else:
              if not labels[int(j)] in res['test'].keys():
                res['test'][labels[int(j)]] = find_char_for_label(all_labels, labels[int(k)])
              else:
                res['test'][labels[int(j)]] += find_char_for_label(all_labels, labels[int(k)])

      # print(cluster.keys())

  for mode in ['test']:
    print()
    print('{} label clustering'.format(mode))
    res[mode] = dict(sorted(res[mode].items()))
    csv_list = []
    for k, v in res[mode].items():
      sim = {}
      for i, label in enumerate(all_labels):
        sim[label] = v.count(find_char_for_label(all_labels, label))
      sim = dict(sorted(sim.items(), key=lambda item: item[1], reverse=True))
      total = 0
      for k1, v1 in sim.items():
        total += v1
      
      
      for k1, v1 in sim.items():
        per = (v1/total) * 100
        sim[k1] = str(round(per, 2))
      
      new_sim = {}
      for k1, v1 in sim.items():
        new_sim[k1] = v1
      print(total, k, new_sim)

      template = {
        'Label': None,
        'Percent': None,
        'First Similar': None,
        'First Percent': None,
        'Second Similar': None,
        'Second Percent': None,
        'Third Similar': None,
        'Third Percent': None,
      }
      i = 0
      r = copy.deepcopy(template)
      for label, percent in new_sim.items():
        if i > 4:
          break
        if i == 0:
          r['Label'] = label
          r['Percent'] = percent
        elif i == 1:
          r['First Similar'] = label
          r['First Percent'] = percent
        elif i == 2:
          r['Second Similar'] = label
          r['Second Percent'] = percent
        elif i == 3:
          r['Third Similar'] = label
          r['Third Percent'] = percent
        i += 1
        
      csv_list.append(r)
      
    print(csv_list)
    generate_csv(name+'.sim', csv_list)

def find_post_train_improvement(name):
  res = {}
  res['train'] = {}
  res['test'] = {}

  def get_cluster_percents(clusters_line):
    clusters = []
    percents = []
    for t, item in enumerate(clusters_line.split(' {')):
      if t != 0:
        item = '{' + item
      clusters.append(ast.literal_eval(item))

    for cluster in clusters:
      for k in cluster.keys():
        percents.append(float(cluster[k].split('|')[-1]))

    return percents
  

  f = open(name, "r")
  content = f.read()
  content_list = content.splitlines()
  f.close()
  max_percents_train = []
  max_percents_test = []
  max_percents_before = []
  max_percents_after = []

  for i, line in enumerate(content_list):
    if not line:
      continue
    train = False

    if line[0] == 'T' or line[0] == "'":
      if line[0] == 'T':
        labels = line[1:].split(', ')
      else:
        labels = line.split(', ')
      
      percents_before = get_cluster_percents(content_list[i+1])
      max_percents_before.append(max(percents_before))

      percents_after = get_cluster_percents(content_list[i+2])
      max_percents_after.append(max(percents_after))

  print(len(max_percents_before))
  print(len(max_percents_after))

  for i, percent in enumerate(max_percents_before):
    if percent > max_percents_after[i]:
      print(percent, max_percents_after[i])


def generate_score_csv_directory(name, n_item=1):
  res = {}
  res['train'] = {}
  res['test'] = {}
  train_list = None
  train_lists = {}
  train_lists['train'] = {}
  train_lists['test'] = {}



  f = open(name, "r")
  content = f.read()

  content_list = content.splitlines()
  f.close()
  all_labels = ["'Web_XSS'", "'DOS_Hulk'", "'attack_portscan'", "'DOS_GoldenEye'", \
   "'DOS_SlowLoris'", "'FTPPatator'", "'attack_DDOS'", "'DOS_SlowHttpTest'",\
   "'attack_bot'", "'Web_BruteForce'", "'SSHPatator'"]

  res = {
     'train': {
       'before': {},
       'after': {}
     },
     'test': {
       'before': {},
       'after': {}
     }
  }

  for i, line in enumerate(content_list):
    if not line:
      continue

    mode = None
    
    if line[0] == 'T' or line[0] == "'":
      if line[0] == 'T':
        labels = line[1:].split(', ')
        mode = 'train'
        train_list = labels
        # if i > 0:
        #   break
      else:
        labels = line.split(', ')
        mode = 'test'
      

      def get_1st_dict(labels, content_line):
        res = {}
        clusters = []
        for t, item in enumerate(content_line.split(' {')):
          if t != 0:
            item = '{' + item
          
          clusters.append(ast.literal_eval(item))
        
        
        for cluster in clusters:
          for k, v in cluster.items():
            if labels[int(k)] not in res.keys():
              res[labels[int(k)]] = []
            res[labels[int(k)]].append(v.split('|')[n_item])
        
        return res

      before_dict = get_1st_dict(labels, content_list[i+1])
      after_dict = get_1st_dict(labels, content_list[i+2])

      for k, v in before_dict.items():
        if k not in res[mode]['before'].keys():
          res[mode]['before'][k] = []
        res[mode]['before'][k].append(v)

        if k not in train_lists[mode].keys():
          train_lists[mode][k] = []
        train_lists[mode][k].append(train_list)

      for k, v in after_dict.items():
        if k not in res[mode]['after'].keys():
          res[mode]['after'][k] = []
        res[mode]['after'][k].append(v)

  # print(len(res['test']['after']))
    
  csv_dict = {
    'Label': None,
    'TrainList': None,
    'N': None,
    'Cluster1': None,
    'Cluster2': None,
    'Cluster3': None,
    'Cluster4': None,
    'Cluster5': None,
    'Cluster6': None,
    'Cluster7': None,
    'Cluster8': None,
  }

  for mode in ['train', 'test']:
    for label in all_labels:
      csv_list = []
      for i, x in enumerate(res[mode]['before'][label]):
        r = copy.deepcopy(csv_dict)
        r['Label'] = label
        r['N'] = len(x)
        r['TrainList'] = train_lists[mode][label][i]
        x.sort(key = float, reverse=True)
        for j, k in enumerate(x):
          r['Cluster'+str(j+1)] = k
        csv_list.append(r)

        r = copy.deepcopy(csv_dict)
        r['Label'] = label
        r['N'] = len(res[mode]['after'][label][i])
        r['TrainList'] = train_lists[mode][label][i]
        res[mode]['after'][label][i].sort(key = float, reverse=True)
        for j, k in enumerate(res[mode]['after'][label][i]):
          r['Cluster'+str(j+1)] = k
        csv_list.append(r)

      model = name.split('/')[-1].split('-')[0]
      filename = '{}-clustering/{}-{}-{}'.format(model, mode, label, n_item)
      print(filename)
      generate_csv(filename, csv_list)


def generate_scores_csv(clustering_folders, score='completeness'):
  import os
  res = {}
  res['train'] = {}
  res['test'] = {}

  skip_files_id = 0 if score == 'completeness' else 1

  for folder_name in clustering_folders:
    file_names = os.listdir(folder_name)
    model_name = folder_name.split('-')[0]

    for file_name in file_names:
      if str(skip_files_id) in file_name:
        continue
      mode = file_name.split("-")[0]
      label = file_name.split("'")[1]
      if label not in res[mode].keys():
        res[mode][label] = {}
      if model_name not in res[mode][label].keys():
        res[mode][label][model_name] = [0, 0]

      
      import csv
      with open('{}/{}'.format(folder_name, file_name), 'r') as file:
          reader = csv.reader(file)
          before_post_train = 0

          for row in reader:
            before_post_train = 1 - before_post_train
            if row[0] == 'Label':
              continue

            for i, cell in enumerate(row):
              percents = row[3:3+int(row[2])]
              percents = [float(x) for x in percents]

            # print(label, percents)

            # To Change the completeness calculation if needed
            if score == 'completeness':
              completeness = max(percents)

              if res[mode][label][model_name][before_post_train] == 0:
                res[mode][label][model_name][before_post_train] = completeness
              else:
                res[mode][label][model_name][before_post_train] = (res[mode][label][model_name][before_post_train] + completeness) / 2
            elif score == 'homogenity':
              homogenity = max(percents)
              
              if res[mode][label][model_name][before_post_train] == 0:
                res[mode][label][model_name][before_post_train] = homogenity
              else:
                res[mode][label][model_name][before_post_train] = (res[mode][label][model_name][before_post_train] + homogenity) / 2
            

  import pprint
  pp = pprint.PrettyPrinter(indent=4)
  pp.pprint(res)
  # print(res)

  template = {
    'Label': None,
    'Mode': None,
    'DOC++': None,
    'DOC++_Post': None,
    'DOC': None,
    'DOC_Post': None,
    'OpenMax': None,
    'OpenMax_Post': None,
  }


  csv_list = []
  for k, v in res.items():
    mode = k
    for label, models in v.items():
      r = copy.deepcopy(template)
      r['Label'] = label
      r['Mode'] = mode
      r['DOC++'] = models['doc++'][0]
      r['DOC++_Post'] = models['doc++'][1]
      r['DOC'] = models['doc'][0]
      r['DOC_Post'] = models['doc'][1]
      r['OpenMax'] = models['openmax'][0]
      r['OpenMax_Post'] = models['openmax'][1]
      csv_list.append(r)
  
  filename = score
  generate_csv(filename, csv_list)


names = [
  'res/doc++-clustering.post',
  'res/openmax-clustering.post',
  'res/doc-clustering.post',
]


# for name in names:
#   extract_clusters(name)

# for name in names:
#   find_post_train_improvement(name+'.list')

for name in names:
  find_similiars(name + '.list')

# for name in names:
#   generate_score_csv_directory(name + '.list', 0)

# for name in names:
#   generate_score_csv_directory(name + '.list', 1)

# clustering_folders = [
#   'doc++-clustering',
#   'doc-clustering',
#   'openmax-clustering',
# ]

# generate_scores_csv(clustering_folders, score='completeness')
# generate_scores_csv(clustering_folders, score='homogenity')