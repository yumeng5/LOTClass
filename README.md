# LOTClass

The source code used for [**Text Classification Using Label Names Only: A Language Model Self-Training Approach**](https://arxiv.org/abs/2010.07245), published in EMNLP 2020.

## Requirements

At least one GPU is required to run the code.

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

Python 3.6 or above is strongly recommended; using older python versions might lead to package incompatibility issues.

## Reproducing the Results

We provide four ```get_data.sh``` scripts for downloading the datasets used in the paper under ```datasets``` and four training bash scripts [```agnews.sh```](agnews.sh), [```dbpedia.sh```](dbpedia.sh), [```imdb.sh```](imdb.sh) and [```amazon.sh```](amazon.sh) for running the model on the four datasets.

**Note: Our model does not use training labels; we provide the training/test set ground truth labels only for completeness and evaluation.**

The training bash scripts assume you have two 10GB GPUs. If you have different number of GPUs, or GPUs of different memory sizes, refer to [the next section](#command-line-arguments) for how to change the following command line arguments appropriately (while keeping other arguments unchanged): ```train_batch_size```, ```accum_steps```, ```eval_batch_size``` and ```gpus```.

## Command Line Arguments

The meanings of the command line arguments will be displayed upon typing
```
python src/train.py -h
```
The following arguments directly affect the performance of the model and need to be set carefully:

* ```train_batch_size```, ```accum_steps```, ```gpus```: These three arguments should be set together. You need to make sure that the **effective training batch size**, calculated as ```train_batch_size * accum_steps * gpus```, is around **128**. For example, if you have 4 GPUs, then you can set ```train_batch_size = 32, accum_steps = 1, gpus = 4```; if you have 1 GPU, then you can set ```train_batch_size = 32, accum_steps = 4, gpus = 1```. If your GPUs have different memory sizes, you might need to change ```train_batch_size``` while adjusting ```accum_steps``` and ```gpus``` at the same time to keep the **effective training batch size** around **128**.
* ```eval_batch_size```: This argument only affects the speed of the algorithm; use as large evaluation batch size as your GPUs can hold.
* ```max_len```: This argument controls the maximum length of documents fed into the model (longer documents will be truncated). Ideally, ```max_len``` should be set to the length of the longest document (```max_len``` cannot be larger than ```512``` under BERT architecture), but using larger ```max_len``` also consumes more GPU memory, resulting in smaller batch size and longer training time. Therefore, you can trade model accuracy for faster training by reducing ```max_len```.
* ```mcp_epochs```, ```self_train_epochs```: They control how many epochs to train the model on masked category prediction task and self-training task, respectively. Setting ```mcp_epochs = 3, self_train_epochs = 1``` will be a good starting point for most datasets, but you may increase them if your dataset is small (less than ```100,000``` documents).

Other arguments can be kept as their default values.

## Running on New Datasets

To execute the code on a new dataset, you need to 

1. Create a directory named ```your_dataset``` under ```datasets```.
2. Prepare a text corpus ```train.txt``` (one document per line) under ```your_dataset``` for training the classification model (no document labels are needed).
3. Prepare a label name file ```label_names.txt``` under ```your_dataset``` (each line contains the label name of one category; if multiple words are used as the label name of a category, put them in the same line and separate them with whitespace characters).
4. (Optional) You can choose to provide a test corpus ```test.txt``` (one document per line) with ground truth labels ```test_labels.txt``` (each line contains an integer denoting the category index of the corresponding document, index starts from ```0``` and the order must be consistent with the category order in ```label_names.txt```). If the test corpus is provided, the code will write classification results to ```out.txt``` under ```your_dataset``` once the training is complete. If the ground truth labels of the test corpus are provided, test accuracy will be displayed during self-training, which is useful for hyperparameter tuning and model cherry-picking using a small test set.
5. Run the code with appropriate command line arguments (I recommend creating a new bash script by referring to the four example scripts).
6. The final trained classification model will be saved as ```final_model.pt``` under ```your_dataset```.

**Note: The code will cache intermediate data and model checkpoints as .pt files under your dataset directory for continued training. If you change your training corpus or label names and re-run the code, you will need to first delete all .pt files to prevent the code from loading old results.**

You can always refer to the example datasets when preparing your own datasets.

## Citations

Please cite the following paper if you find the code helpful for your research.
```
@inproceedings{meng2020text,
  title={Text Classification Using Label Names Only: A Language Model Self-Training Approach},
  author={Meng, Yu and Zhang, Yunyi and Huang, Jiaxin and Xiong, Chenyan and Ji, Heng and Zhang, Chao and Han, Jiawei},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year={2020},
}
```
