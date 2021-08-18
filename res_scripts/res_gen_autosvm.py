import ast
import copy

name = 'newRes/autoSvm'

f = open(name+".log", "r")
content = f.read()

content_list = content.splitlines()
f.close()

expriments = []
midlines = []


def extract_data(expriment_lines):
    data = {}
    # data['wrong_classified'] = {}
    # data['acc'] = {}
    # data['total'] = {}

    data['train_list'] = None
    data['test_list'] = []
    data['test_res'] = []
    # data['test_res_after'] = []

    for i, line in enumerate(expriment_lines):
        if 'Training on' in line:
            a = line.index('[')
            b = line.index(']')
            data['train_list'] = line[a+1:b]
        
        if 'Validate' in line:
            # print(expriment_lines)
            a = line.index('[')
            b = line.index(']')
            data['test_list'].append(line[a+1:b])

            data['test_res'].append(ast.literal_eval(expriment_lines[i+1]))
            # data['test_res_after'].append(ast.literal_eval(expriment_lines[i+3]))
            

            # valid_split = line.split('[')
            
            # labels = valid_split[1][:-1].split(',')
            # labels = [label.strip()[1:-1] for label in labels]
            # data['labels'] = labels
        # else:
        #     res = ast.literal_eval(line)
        #     for l in res.keys():
        #         data['wrong_classified'][l] = res[l][3]
        #         data['acc'][l] = res[l][2]
        #         data['total'][l] = res[l][0]
    # print(data)
    return data

def parse_data(data):
    template = {
        'train_list': None,
        'test1': None,
        'n1': None,
        'acc1': None,
        'unknowns1': None,
        'test2': None,
        'n2': None,
        'acc2': None,
        'unknowns2': None,
        'test3': None,
        'n3': None,
        'acc3': None,
        'unknowns3': None,
        'test4': None,
        'n4': None,
        'acc4': None,
        'unknowns4': None,
        'test5': None,
        'n5': None,
        'acc5': None,
        'unknowns5': None,
    }
    res_list = []

    for n, test in enumerate(data['test_list']):
        r = copy.deepcopy(template)
        r['train_list'] = data['train_list']
        test_labels = data['test_list'][n]
        test_labels = [item[1:-1] for item in test_labels.split(', ')]
        res = data['test_res'][n]
        # res_after = data['test_res_after'][n]
        # del res_after['n_unkowns']

        # print(test_labels)
        # print(res_before)
        # print(res_after)
        # exit()
        # print(test_labels)
        for i in range(5):
            r['test'+ str(i+1)] = test_labels[i]
            r['n'+ str(i+1)] = res[i][0]
            r['acc' + str(i+1)] = res[i][2]
            # r['acc' + str(i+1) + '_1'] = res_after[i][2]
            r['unknowns' + str(i+1)] = res[i][3]
            # r['unknowns' + str(i+1) + '_1'] = res_after[i][3]

        res_list.append(r)
    return res_list


for i, line in enumerate(content_list):
    if midlines and ('Training on' in line or i == len(content_list)):
        expriments.append(midlines)
        midlines = []
    midlines.append(line)

# expriments = expriments[1:]
# print(len(expriments))
# print(expriments[1])
# data = extract_data(expriments[1])
# for key in data.keys():
#     print(key, data[key])
# print(len(data['test_list']))
# print(len(data['test_res_before']))
# print(len(data['test_res_after']))
# res = parse_data(data)

datas = []
for expriment in expriments:
    data = extract_data(expriment)
    datas.append(data)
# print(len(datas))

results = []
for data in datas:
    res = parse_data(data)
    results.extend(res)


# print(len(results))

# exit()
import csv
def generate_csv(results):
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

generate_csv(results)