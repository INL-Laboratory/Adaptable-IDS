import csv

def parse_line(data_line):
  res = {
    'label': data_line['test5'],
    'similairs': [],
    'acc': float(data_line['acc5_1']),
    'acceptedExp': 0
  }
  unknown_label = data_line['test5']
  unknown_n = int(data_line['n5'])
  
  badExp = False

  for i in range(1, 6):
    n = int(data_line['unknowns'+ str(i) + '_1'])
    # if float(data_line['acc5_1']) < 

    if n > unknown_n*0.3:
      badExp = True
      res['similairs'].append(data_line['test'+str(i)])
  
  if not badExp:
    res['acceptedExp'] = 1
  return res
    


results = {}

with open('./res/doc_noNormal.csv', newline='') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    res = parse_line(row)
    # print(res)
    label = res['label']
    # print(label)
    if not label in results.keys():
      results[label] = {
        'acc': [],
        'similairs': [],
        'acceptedExp': 0,
        'count': 0
      }
    # print(results)
    acceptedExp = res['acceptedExp']
    results[label]['acceptedExp'] += acceptedExp
    results[label]['count'] += 1
    results[label]['acc'].append(res['acc'])
    results[label]['similairs'].extend(res['similairs'])



import statistics

for i in results.keys():
  count = {x: results[i]['similairs'].count(x) for x in results[i]['similairs']}

  print(i, count)

  print(statistics.mean(results[i]['acc']), statistics.pstdev(results[i]['acc']))
  print(results[i]['acceptedExp'], results[i]['count'])
  print()
# print(results)
