# MODE-LSTM
The repository for EMNLP 2020 accepted paper "MODE-LSTM: A Parameter-efficient Recurrent Network with Multi-Scale for Sentence Classification"

####  Dependencies

* python 2.7
* tensorflow 1.14.0
* keras 2.2.4

####  Data Preprocessing

The folder `data` contains the test dataset `SST5`, the dataset `IE` can be downloaded according to paper "Dynamic Compositional Neural Networks over Tree Structure", and the remaining data sets can refer to repository [TextCNN](https://github.com/yoonkim/CNN_sentence).

First, you should download the pretrain word embedding `glove.840B.300d.txt` from [glove](https://nlp.stanford.edu/projects/glove/), and place it under folder `data`.

To process the raw data `SST5`, run 

````
python text_process.py
````

####  Running model

You can run the command

```python
python modelstm.py
```

####  Reference

```
@inproceedings{ma-etal-2020-mode,
    title = "{MODE}-{LSTM}: A Parameter-efficient Recurrent Network with Multi-Scale for Sentence Classification",
    author = "Ma, Qianli  and
      Lin, Zhenxi  and
      Yan, Jiangyue  and
      Chen, Zipeng  and
      Yu, Liuhong",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    pages = "6705--6715"
}
```




