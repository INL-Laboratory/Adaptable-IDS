# Adaptable-IDS


### DOC, DOC++, OpenMax
For DOC++, DOC, and OpenMax methods, use the following command:
```
python3 main.py --arch CNN --validation_mode OpenMax --loss_function softmax --target classification
```

The available options are:
```bash
--arch: CNN, LSTM                                --- default is CNN
--validation_mode: DOC, DOC++, OpenMax, CROSR    --- default is DOC++
--loss_function: 1-vs-rest, cross-entropy        --- default is 1-vs-rest
--target: classification, clustering             --- default is classification
```


### AutoSVM
The AutoSVM codes are available in ```auto_svm``` directory.

```bash
python3 -m auto_svm.main
```

### Results
```res_scripts``` directory contains all the required scripts to generate csvs from log files.


```
res_gen.py -- Generates the classification accuracy csv for DOC, DOC++, OpenMax

res_gen_autosvm.py -- Generates the classification accuracy csv for AutoSVM

accepted_experiments.py -- Calculate accepted experiments and similar labels based on output of last two scripts

clustering.py -- Implements different clustering analysis functions:
1) extract_clusters: Takes a clustering log file as input and generates a .list file containing clusters of all experiments.
2) find_post_train_improvement: Prints all the completeness improvements for experiments based on a .list file.
3) find_similiars: Find similarities based on clustering using .list file.
4) generate_score_csv_directory: Genrates a directory contaning a csv for completeness or homogenity of each experiment based on the .list file
5) generate_score_csv: Generate a single csv for completeness or homogenity based on the generated directory
```


### WIP
Add the results directory