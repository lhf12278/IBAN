# Intermediary-guided Bidirectional Spatial-Temporal Aggregation Network for Video-based Visible-Infrared Person Re-Identification

## Pipeline

![framework](./1.pdf)


## Requirements

### Installation

We use /torch >=1.8 / 24G  RTX3090 for training and evaluation.

### Prepare Datasets
mkdir data_original

mkdir data_anaglyph

There are many ways to generate anaglyph images, and you can also use the code (**main_VCM.py**) we provide.

Note that the organization, file name, and storage format of the original data and the anaglyph data should be consistent.

```
data
├── data_original
│   └── 
│   └── 
│   └── 
│   └── ..
├── data_anaglyph
│   └── 
│   └── 
│   └── 
│   └── ..
```

## Training and Evaluation

```shell
python train.py
```

Later, we will upload our trained model([download](https://drive.google.com/file/d/1DaMfPMzvW2kO6YhxCBahLma0dtPO4CZG/view?usp=share_link)), and you can load the model directly without training.


## Contact

If you have any questions, please feel free to contact me. ( liuminghui_1997@163.com ).
