# LOTClass

The source code used for **Text Classification Using Label Names Only: A Language Model Self-Training Approach**, published in EMNLP 2020.

## Requirements

At least one GPU is required to run the code.

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

Python 3.6 is strongly recommended; using older python versions might lead to package incompatibility issues.

## Reproducing the Results

We provide the four datasets used in the paper under ```datasets``` and four bash scripts ```agnews.sh```, ```dbpedia.sh```, ```imdb.sh``` and ```amazon.sh``` for running the model on the four datasets.

The bash scripts assume you have two 10GB GPUs. If you have different number of GPUs, or GPUs of different memory sizes, refer to for how to set command line arguments appropriately.

## Inputs

The weak supervision sources ```${sup_source}``` can come from any of the following:
* Label surface names (```labels```); you need to provide class names for each class in ```./${dataset}/classes.txt```, where each line begins with the class id (starting from ```0```), followed by a colon, and then the class label surface name. 
* Class-related keywords (```keywords```); you need to provide class-related keywords for each class in ```./${dataset}/keywords.txt```, where each line begins with the class id (starting from ```0```), followed by a colon, and then the class-related keywords separated by commas. 
* Labeled documents (```docs```); you need to provide labeled document ids for each class in ```./${dataset}/doc_id.txt```, where each line begins with the class id (starting from ```0```), followed by a colon, and then document ids in the corpus (starting from ```0```) of the corresponding class separated by commas. 

Examples are given under ```./agnews/``` and ```./yelp/```.

## Outputs

The final results (document labels) will be written in ```./${dataset}/out.txt```, where each line is the class label id for the corresponding document.

Intermediate results (e.g. trained network weights, self-training logs) will be saved under ```./results/${dataset}/${model}/```.

## Running on a New Dataset

To execute the code on a new dataset, you need to 

1. Create a directory named ```${dataset}```.
2. Put raw corpus (with or without true labels) under ```./${dataset}```.
3. Modify the function ```read_file``` in ```load_data.py``` so that it returns a list of documents in variable ```data```, and corresponding true labels in variable ```y``` (If ground truth labels are not available, simply return ```y = None```).
4. Modify ```main.py``` to accept the new dataset; you need to add ```${dataset}``` to argparse, and then specify parameter settings (e.g. ```update_interval```, ```pretrain_epochs```) for the new dataset.

You can always refer to the example datasets when adapting the code for a new dataset.

## Citations

Please cite the following paper if you find the code helpful for your research.
```
@inproceedings{meng2018weakly,
  title={Weakly-Supervised Neural Text Classification},
  author={Meng, Yu and Shen, Jiaming and Zhang, Chao and Han, Jiawei},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={983--992},
  year={2018},
  organization={ACM}
}
```
