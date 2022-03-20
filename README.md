# Importance-Weighted Domain Alignment

Modules for performing class-importance-weighted domain alignment (IWDA) in PyTorch for unsupervised domain adaptation, as well as on text and token classifiers built upon pre-trained language models using the [🤗 Transformers library](https://github.com/huggingface/transformers). The alignment is implemented using adversarial training with Wasserstein-1 critic loss and zero-centered gradient penalty.

The modules are found in `domain_alignment.py`. They are `W1CriticWithImWeights`, `W1CriticWithImWeightsEstimation`, and `ImWeightsEstimator`. See `run_text_cls.py`, `run_token_cls.py` and below for example usage.

## Example: Unsupervised Cross-Lingual Learning

One usage of IWDA is unsupervised cross-lingual transfer of pre-trained language models on downstream tasks, discussed and evaluated in our ICLR 2022 paper “[Cross-Lingual Transfer with Class-Weighted Language-Invariant Representations](https://openreview.net/forum?id=k7-s5HSSPE5)”. 

The commands for the following unsupervised cross-lingual transfer tasks are provided in `run_iclr2022.sh` (also check package `requirements.txt`):

- mBERT transfer from English to Spanish for named-entity recognition on CoNLL-2002 and 2003 datasets.
- mBERT transfer from English to Japanese for sentiment analysis on Multilingual Amazon Reviews Corpus.

Results on CoNLL NER with mBERT (average of 5 runs):

| Method        | de    | es    | nl    |
| ------------- | ----- | ----- | ----- |
| Zero-shot     | 69.77 | 74.14 | 78.28 |
| IWDA          | 72.56 | 76.11 | 78.63 |
| IWDA (oracle) | 72.58 | 76.48 | 79.17 |

Results on MARC sentiment analysis with mBERT (average of 3 runs):

| Method        | de    | es    | fr    | ja    | zh    |
| ------------- | ----- | ----- | ----- | ----- | ----- |
| Zero-shot     | 44.80 | 46.49 | 46.02 | 37.37 | 38.48 |
| IWDA          | 51.94 | 49.77 | 49.78 | 42.62 | 44.04 |
| IWDA (oracle) | 51.95 | 50.83 | 50.01 | 44.91 | 45.96 |

## References

The BibTeX entry for our paper is:

```bibtex
@inproceedings{xian2022crosslingual,
  title={Cross-Lingual Transfer with Class-Weighted Language-Invariant Representations},
  author={Ruicheng Xian and Heng Ji and Han Zhao},
  year={2022},
  booktitle={International Conference on Learning Representations},
  url={https://openreview.net/forum?id=k7-s5HSSPE5}
}
```

Our implementation follows the following paper closely, where some code [come from](https://github.com/microsoft/Domain-Adaptation-with-Conditional-Distribution-Matching-and-Generalized-Label-Shift):

```bibtex
@inproceedings{tachetdescombes2020domainadaptation,
  title={Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift},
  author={Tachet des Combes, Remi and Zhao, Han and Wang, Yu-Xiang and Gordon, Geoff},
  year={2020},
  booktitle={Advances in Neural Information Processing Systems}
}
```
