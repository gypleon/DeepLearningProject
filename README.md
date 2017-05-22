~~Sentence-based Emotion Recognition on Evolutionary Neural Networks~~

Evolve Complex Neural Networks for Natural Language Model
=====
# Introduction
The final project for ENGG5189 Advanced Topics in Artificial Intelligence at the Chinese University of Hong Kong.

Baseline network structure is designed by [Yoon Kim](https://github.com/yoonkim/lstm-char-cnn) and neural network codes are partially borrowed from [mkroutikov](https://github.com/mkroutikov/tf-lstm-char-cnn).

# Dataset
[Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42)

# Requirement
* Ubuntu 14.04 LTS
* Python 3.4.3 -> Python 2.7.6
* Tensorflow 1.0.0 -> Tensorflow 1.0.1

# Usage
## perform evolution
python evolution.py
## results
The evolution procedure would be printed on stdout including network structures of individuals at every generation. Other information, e.g. winners and losers at each generation, are also printed.

# Acknoledgement
I am very grateful to the paper written by Yoon Kim and the codes from mkroutikov. Special thanks should be given to [Prof. LEUNG, Kwong-Sak](http://www.cs.cuhk.edu.hk/~ksleung/) and the tutor [LIU Pengfei](https://scholar.google.com.hk/citations?hl=en&view_op=list_works&gmla=AJsN-F7ES3mHLxTANgDceXsyYFXLlCm89-AxyODSAFmmHYwsbOUzVY169qXqlgozcpk6JBmvDXMgVi3bT26sxJlu6BIrnq3eZA&user=Jr-faBMAAAAJ) who give me numerous help.

# Reference
1. [Kim, Yoon, et al. "Character-aware neural language models." arXiv preprint arXiv:1508.06615 (2015).](https://arxiv.org/abs/1508.06615)
2. [Marcus, Mitchell P., Mary Ann Marcinkiewicz, and Beatrice Santorini. "Building a large annotated corpus of English: The Penn Treebank." Computational linguistics 19.2 (1993): 313-330.](http://dl.acm.org/citation.cfm?id=972475)
