## XOR Mixup &mdash; TensorFlow experiment codes (FL'ICML 2020)
### : Privacy-Preserving Data Augmentation for One-Shot Federated Learning

### [Paper](https://arxiv.org/abs/2006.05148) | [Presentation](https://github.com/ihooni/XOR-Mixup/files/5049557/FL_ICML20_poster_session-final.pdf)

> **XOR Mixup: Privacy-Preserving Data Augmentation for One-Shot Federated Learning**<br>
> MyungJae Shin (SNUH), Chihoon Hwang (CAU), Joongheon Kim (Korea), Jihong Park(Deakin), Mehdi Bennis(Oulu), Seong-Lyun Kim(Yonsei)
>
> **Abstract** *User-generated data distributions are often imbalanced across devices and labels, hampering the performance of federated learning (FL). To remedy to this non-independent and identically distributed (non-IID) data problem, in this work we develop a privacy-preserving XOR based mixup data augmentation technique, coined XorMixup, and thereby propose a novel one-shot FL framework, termed XorMixFL. The core idea is to collect other devices' encoded data samples that are decoded only using each device's own data samples. The decoding provides synthetic-but-realistic samples until inducing an IID dataset, used for model training. Both encoding and decoding procedures follow the bit-wise XOR operations that intentionally distort raw samples, thereby preserving data privacy. Simulation results corroborate that XorMixFL achieves up to 17.6% higher accuracy than Vanilla FL under a non-IID MNIST dataset.*

## Citation
```
@article{shin2020xor,
  title={XOR Mixup: Privacy-Preserving Data Augmentation for One-Shot Federated Learning},
  author={Shin, MyungJae and Hwang, Chihoon and Kim, Joongheon and Park, Jihong and Bennis, Mehdi and Kim, Seong-Lyun},
  journal={arXiv preprint arXiv:2006.05148},
  year={2020}
}
```

## Author
[MyungJae Shin](https://github.com/170928), Chihoon Hwang
