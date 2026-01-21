# The Official PyTorch Implementation of aL-SAR (adaptive Layer freezing and Similarity-Aware Retrieval)
<a href="https://openreview.net/pdf?id=dOAkHmsjRX"> <b>Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling</b> </a>
<br>
<a href="https://dbd05088.github.io/">Minhyuk Seo*</a>,
Hyunseo Koh*,
<a href="http://ppolon.github.io/"> Jonghyun Choi </a>
<br>
<a href="https://iclr.cc/"> ICLR 2025 (Spotlight) </a>


We provide the official implementation of the proposed aL-SAR and baselines.


## Environment
### Clone repository
```
git clone https://github.com/snumprlab/budgeted-cl.git
```

### Training environment
```
conda create -n budgeted_cl python=3.10
conda activate budgeted_cl
pip install -r requirements.txt
```

### Downloading the Datasets
CIFAR-10/100, CLEAR100, Bongard-HOI, and Bongard-OpenWorld can be downloaded by running the corresponding scripts in the `dataset/` directory.
ImageNet dataset can be downloaded from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge).


## Training


### Experiments Using Shell Script
First, activate the training environment `budgeted_cl`.
```
conda deactivate
conda activate budgeted_cl
```

Experiments for the implemented methods can be run by executing `train_VLM.sh` by
<pre>
bash train_VLM.sh
</pre>
You may change various arguments for different experiments.
- `NOTE`: Short description of the experiment. Experiment result and log will be saved at `results/DATASET/NOTE`.
  - WARNING: logs/results with the same dataset and note will be overwritten!
- `MODEL_ARCH`: VLM model types. Supported VLMs are: [llava, bunny_3b, bunny_8b]
- `DATASET`: Dataset to use in experiment. Supported datasets for multi-modal concept-IL are: [Bongard-HOI, Bongard-OpenWorld] Supported datasets for CIL are: [cifar10, cifar100, clear10, clear100, imagenet]
- `MEM_SIZE`: Maximum number of samples in the episodic memory.
- `NUM_ITER`: Number of model updates per sample.
- `NUM_SET`: Number of samples in each positive and negative support set for Bongard benchmarks.
- `RND_SEED`: Random seed for reproducibility.
- `DATA_TYPE`: Data source type. The default is manually annotated data. Supported types: [ma_ver3_more_text, generated, web].
- `Ours`: Enable our proposed adaptive layer-freezing (aL).
- `SAR`: Enable our proposed Similarity-aware Retrieval (SAR).

**Note**: All hyperparameters used for the experiments in the paper are set as default.


## Evaluation
First, activate the evaluation environment `budgeted_cl`.
```
conda deactivate
conda activate budgeted_cl
```

To evaluate a model, run `eval_VLM.sh` by
<pre>
bash train_VLM.sh
</pre>

You may change various arguments for different experiments.
- `RND_SEED`: Random seed (must be kept the same during training).
- `NOTE`: Short description of the experiment (must be kept the same during training to preserve the model path directory).
- `NUM_SET`: Number of samples in each positive and negative support set for Bongard benchmarks.



## License
GNU GENERAL PUBLIC LICENSE


## Citation
**aL-SAR**
```
@inproceedings{
  seo2025budgeted,
  title={Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling},
  author={Minhyuk Seo and Hyunseo Koh and Jonghyun Choi},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=dOAkHmsjRX}
}
```

## Acknowlegment
```
This work was partly supported by AI Center, Samsung Electronics, and the IITP grants (No.RS2022-II220077, No.RS-2022-II220113, No.RS-2022-II220959, No.RS-2021-II211343 (SNU AI),
No.RS-2021-II212068 (AI Innov. Hub)) funded by the Korea government (MSIT).
```
