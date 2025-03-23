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

### Install PyTorch
Install PyTorch from <a href="https://pytorch.org/get-started/previous-versions/#v1100">the official PyTorch site</a> for both `cl-alfred-train` and `cl-alfred-eval`.
```
conda deactivate
conda activate cl-alfred-train
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

conda deactivate
conda activate cl-alfred-eval
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### Downloading the Datasets
CIFAR10, CIFAR100, CLEAR10, CLEAR100, Bongard-HOI, and Bongard-OpenWorld can be downloaded by running the corresponding scripts in the `dataset/` directory.
ImageNet dataset can be downloaded from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge).


## Training


### Experiments Using Shell Script
First, activate the training environment `cl-alfred-train`.
```
conda deactivate
conda activate cl-alfred-train
```

Experiments for the implemented methods can be run by executing `ex.sh` by
<pre>
bash ex.sh
</pre>
You may change various arguments for different experiments.
- `NOTE`: Short description of the experiment. Experiment result and log will be saved at `results/DATASET/NOTE`.
  - WARNING: logs/results with the same dataset and note will be overwritten!
- `MODE`: CL method to be applied. Methods implemented in this version are: [sdp, clib, er, mir, gdumb, der++]
- `DATASET`: Dataset to use in experiment. Supported datasets are: [cifar10, cifar100, tinyimagenet, imagenet]
- `REPEAT`: Number of periods in the Periodic Gaussian data stream. Set `REPEAT=1` for non-periodic case.
- `SIGMA`: Standard deviation of the Gaussian distribution in Gaussian and Periodic Gaussian data stream.
- `USE_AMP`: Use automatic mixed precision (amp), for faster running and reducing memory cost.
- `MEM_SIZE`: Maximum number of samples in the episodic memory.
- `ONLINE_ITER`: Number of model updates per sample.
- `EVAL_PERIOD`: Period of evaluation queries, for calculating <img src="https://render.githubusercontent.com/render/math?math=A_\text{AUC}">.
- `F_PERIOD`: Period of evaluating knowledge gain/loss, for calculating KLR and KGR.

```
python models/train/train_seq2seq.py        \
    --incremental_setup <incremental_setup> \
    --mode <mode>                           \
    --stream_seed <stream_seed>             \
    --dout <path_to_save_weight>
```
**Note**: All hyperparameters used for the experiments in the paper are set as default.

For example, if you want train `CAMA` for the `Behavior-IL` setup with a stream seed `1` and save the weights in `exp/behavior_il/cama/s1`, the command may look like below.
```
python models/train/train_seq2seq.py        \
    --incremental_setup behavior_il         \
    --mode cama                             \
    --stream_seed 1                         \
    --dout exp/behavior_il/cama/s1
```

## Evaluation
First, activate the evaluation environment `budgeted_cl`.
```
conda deactivate
conda activate budgeted_cl
```

To evaluate a model, run `eval_seq2seq.py` with the hyper-parameters below. <br>
- `model_path`: the path of a saved model
- `eval_split`: `valid_seen` (Valid Seen) or `valid_unseen` (Valid Unseen)
- `x_display`: the number of your display (check `echo $DISPLAY` in your command)
- `incremental_setup`: `behavior_il` (Behavior-IL) or `environment_il` (Environment-IL)
- `incremental_type`: the current incremental type learned
  - Behavior-IL: `look_at_obj_in_light`, `pick_heat_then_place_in_recep`, `pick_two_obj_and_place`, `pick_cool_then_place_in_recep`, `pick_and_place_simple`, `pick_clean_then_place_in_recep`, `pick_and_place_with_movable_recep`
  - Environment-IL: `Kitchen`, `Livingroom`, `Bedroom`, `Bathroom`
- `stream_seed`: the seed number of a random stream (`1`, `2`, `3`, `4`, and `5`)
- `num_threads`: the number of simulators used for evaluation
- `gpu`: the usage of GPU during evaluation
```
python models/eval/eval_seq2seq.py --gpu      \
    --model_path <model_path>                 \
    --eval_split <eval_split>                 \
    --incremental_setup <incremental_setup>   \
    --num_threads <num_threads>               \
    --x_display <x_display>                   \
    --gpu
```
**Note**: All hyperparameters used for the experiments in the paper are set as default.<br>
**Note**: For cpu-only evaluation, remove the option `--gpu`.

If you want to evaluate our model saved in `exp/behavior_il/cama/s1/net_epoch_000002251_look_at_obj_in_light.pth` in the `seen` validation for the current task `look_at_obj_in_light` of the `Behavior-IL` setup trained with a random stream sequence `1`, you may use the command below.
```
python models/eval/eval_seq2seq.py                                                    \
    --model_path exp/behavior_il/cama/s1/net_epoch_000002251_look_at_obj_in_light.pth \
    --eval_split valid_seen                                                           \
    --incremental_setup behavior_il                                                   \
    --incremental_type look_at_obj_in_light                                           \
    --stream_seed 1                                                                   \
    --num_threads 3                                                                   \
    --x_display 1                                                                     \
    --gpu
```
**Note**: Choose your available display number `x_display`.<br>
**Note**: Adjust your thread number based on your system `num_threads`.


## Hardware
Trained and tested on:
- **GPU** - NVIDIA RTX A6000 (48GB)
- **CUDA** - CUDA 12.0
- **CPU** - 12th Gen Intel(R) Core(TM) i7-12700K
- **RAM** - 64GB
- **OS** - Ubuntu 20.04


## License
GNU GENERAL PUBLIC LICENSE


## Citation
**CL-ALFRED**
```
@inproceedings{kim2024online,
  title={Online Continual Learning for Interactive Instruction Following Agents},
  author={Kim, Byeonghwi and Seo, Minhyuk and Choi, Jonghyun},
  booktitle={ICLR},
  year={2024}
}
```
**i-Blurry**
```
@inproceedings{koh2022online,
  title={Online Continual Learning on Class Incremental Blurry Task Configuration with Anytime Inference},
  author={Koh, Hyunseo and Kim, Dahyun and Ha, Jung-Woo and Choi, Jonghyun},
  booktitle={ICLR},
  year={2022}
}
```
**ABP**
```
@inproceedings{kim2021agent,
  author    = {Kim, Byeonghwi and Bhambri, Suvaansh and Singh, Kunal Pratap and Mottaghi, Roozbeh and Choi, Jonghyun},
  title     = {Agent with the Big Picture: Perceiving Surroundings for Interactive Instruction Following},
  booktitle = {Embodied AI Workshop @ CVPR 2021},
  year      = {2021},
}
```
**ALFRED**
```
@inproceedings{ALFRED20,
  title ={{ALFRED: A Benchmark for Interpreting Grounded
           Instructions for Everyday Tasks}},
  author={Mohit Shridhar and Jesse Thomason and Daniel Gordon and Yonatan Bisk and
          Winson Han and Roozbeh Mottaghi and Luke Zettlemoyer and Dieter Fox},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020},
  url  = {https://arxiv.org/abs/1912.01734}
}
```

## Acknowlegment
```
This work was partly supported by the NRF grant (No.2022R1A2C4002300, 15%) and IITP grants (No.2020-0-01361 (10%, Yonsei AI), No.2021-0-01343 (5%, SNU AI), No.2022-0-00077 (10%), No.2022-0-00113 (20%), No.2022-0-00959 (15%), No.2022-0-00871 (15%), No.2021-0-02068 (5%, AI Innov. Hub), No.2022-0-00951 (5%)) funded by the Korea government (MSIT).
```
