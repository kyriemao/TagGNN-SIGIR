# TagGNN

This is Pytorch implementation for our SIGIR 2020 paper:

> Kelong Mao, Xi Xiao, Jieming Zhu, Biao Lu, Ruiming Tang, Xiuqiang He. Item Tagging for Information Retrieval: A Tripartite Graph Neural Network based Approach. [Paper in arXiv](https://arxiv.org/abs/2008.11567).



## Introduction
In this work, we proposed TagGNN, a heterogeneous graph neural network for more accurate item tagging under information retrieval scenario.



## Code
* main.py: All python code to reproduce UltraGCN
* dataset_name_config.ini: The configuration file which includes parameter settings for reproduction on the specific dataset.

```bash
python main.py --config_file dataset_config.ini
```
