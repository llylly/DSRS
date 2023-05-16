# Double Sampling Randomized Smoothing

----
[![BSD license](https://img.shields.io/badge/License-BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Official code repo for [ICML 2022] [Double Sampling Randomized Smoothing](https://arxiv.org/abs/2206.07912).


Double sampling randomized smoothing (DSRS) is a novel robustness certification framework for randomized smoothing. 

**The randomized smoothing (RS)** (c.f. [Cohen et al](https://arxiv.org/abs/1902.02918), [Lecuyer et al](https://arxiv.org/abs/1802.03471)) adds noise to the input and uses majority voting among the model predictions for multiple noised inputs to get the final label prediction.
Since the shifting of the center of noised input distribution does not change the ranking of label predictions, the RS models are certifiably robust to small L2-norm-bounded perturbations.

Given an input instance and a model, robustness certification approach computes a robust radius, such that any perturbations within the radius does not the change the final prediction. By nature, certification approach is conservative --- it provides a lower bound of robust radius while the maximum robust radius for the given input instance could be much larger than the one computed by the certification approach.

For RS, the most widely-used certification approach is called Neyman-Pearson-based certfication. It leverages the probability of base model's predicting each class under the input noise to compute the certification.

**In this work**, we propose to sample the base model's prediction statistics under **two different distributions**, and leverage the joint information to compute the certification. Since we leverage more information, our certification approach is guaranteed to be tighter (if not equal) than the most widely-used Neyman-Pearson-based approach.

![High-Level Illustration of DSRS](readme_fig/overall_pipeline.png)


More details can be found in our paper.

### Usage

##### Preparation

1. Install recommended environment: python 3.9.7, scikit-learn >= 0.24, torch 1.10 with GPU (for fast sampling of DNN models).

2. If you need to reproduce the results from certifying pretrained models (see *Guidelines for Reproducing Experimental Results > Scenario B*), download the trained models from Figshare (https://figshare.com/articles/software/Pretrained_Models_for_DSRS/21313548), and unzip to `models/` folder. After unzip, the `models` folder should contain 5 subfolder: `models/cifar10` (smoothadv's pretrained best CIFAR-10 models), `models/consistency` (reproduced consistency models), `models/new_cohen` (reproduced Cohen's Gaussian augmentation models), `models/salman` (reproduced Salman et al's models), and `models/smoothmix` (reproduced SmoothMix models).

##### Running Certification from Scratch

The main entrance of our code is `main.py`, which computes both the Neyman-Pearson-based certification and DSRS certification given the sampled probability folder path.

Given a input test dataset and a model, to compute the DSRS certification, we need the following **three steps**:

1. **Sampling & get the probability (P_A and Q_A) under two distributions**

`sampler.py` loads the model and does model inference via PyTorch APIs. It will output pA or qA to corresponding txt file in `data/sampling/{model_filename.pth (.tar extension name is trimmed)}/` folder (will create the folder). Note that each run only samples pA from one distribution, so:

- If the end-goal is to compute only the Neyman-Pearson-based certification, we only need one probability (P_A), and thus we run sampler.py just once for the model.

- If the end-goal is to compute the DSRS certification, we need two probabilties from two different distributions (P_A and Q_A), and thus we run sampler.py twice for the model.

Main usage:

`python sampler.py [dataset: mnist/cifar10/imagenet/tinyiamgenet] [model: models/*/*.pth.tar] [sigma] --disttype [gaussian/general-gaussian] {--k [k]} {--th [number between 0 and 1 or "x+" for adaptive thresholding]} --N [sampling number, usually 50000 for DSRS, 100000 for Neyman-Pearson] --alpha [confidence, usually 0.0005 for DSRS, 0.001 for Neyman-Pearson]`


There are other options such as batch size, data output directory, GPU no. specification, etc. Please browse `parser.add_arguments()` statements to get familier with them.

- If the distribution is Gaussian, we don't need to specify $k$.
- If the distribution is generalized Gaussian, we need to specify $k$, whose meaning can be found in the paper.
- If the distribution is Gaussian or generalized Gaussian with thresholding, `--th` specifies the threshold. 
  - If the threshold is a static value, `th` is a real number meaning the percentile. 
  - If the threshold is depended by pA, `th` is "x+" (there are other heuristics but they do not work well), and the script will search the pA file to determine the threshold dynamically.
    

2. **Compute the Neyman-Pearson-based certification**

`main.py` is the entrance for computing both the Neyman-Pearson-based certification and the DSRS certification. Note compute the DSRS certification, one first needs to execute this step, i.e., compute the Neyman-Pearson-based certification, since DSRS tries to increase the radius certified strating from the certified radius of Neyman-Pearson-based certification.

`main.py` is built solely on CPU, mainly relying on scikit-learn package.

Main usage:

`python main.py [dataset: mnist/cifar10/imagenet/tinyimagenet] origin [model: *.pth - will read from data/sampling/model, just type in "*.pth" as the name without relative path] [disttype = gaussian/general-gaussian] [sigma] [N: sampling number, used to index the sampling txt file] [alpha: confidence, used to index the sampling file] {--k [k]}`

There are other options that can customized the folder path of sampling data or the parallelized CPU processes. But the default one is already good. Note that some arguments in `main.py` only have effects when computing DSRS certification, such as `-b`, `--improve_*`, `--new_rad_dir`.

If the distribution type is generalized Gaussian, we need to specify k, otherwise not.

For standard Gaussian, we use the closed-form expression in Cohen et al to compute the certification. 
For generalized Gaussian, we use the numerical integration method in Yang et al to compute the certification.

3. **Compute the DSRS certification**

Once the Neyman-Pearson-based certification is computed, we run `main.py` again but use different arguments to compute the DSRS certification. 

Main usage:

`python main.py [dataset: mnist/cifar10/imagenet/tinyimagenet] improved [model: *.pth - will read from data/sampling/model, just type in "*.pth" as the name without relative path] [disttype = gaussian/general-gaussian/gaussian-th/general-gaussian-th] [sigma] [N: sampling number, used to index the sampling txt file] [alpha: confidence, used to index the sampling file] {--k [k]} {-b b1 b2 ...} {--improve_mode grid/fast/precise} {--improve_unit real_number} {--improve_eps real_number}`

Note that the arguments are different from step 2, where `origin` changed to `improved`. The script will read in the previous Neyman-Pearson-based certification files, and compute the improved certification.

Distribution P's parameters are specified by `disttype` and `k`. Specifically, if `disttype` is `gaussian-th` or `general-gaussian-th`, the P distribution is Gaussian or generalized Gaussian respectively, and the Q distribution is thresholded Gaussian or thresholded generalized Gaussian respectively.

Distribution Q is of the same `disttype` and has the same `k` as P. The difference is in variance (if `disttype` is `gaussian` or `general-gaussian`) or the threshold (if `disttype` is `gaussian-th` or `general-gaussian-th`). The variance or the threshold (real number if static percentile threshold, `x+` if dynamic heuristic based threshold) is specified by `b1`, `b2`, ..., where each `bi` stands for one option of Q distribution, i.e., the script supports computing the certification with one P and multiple different Q's in a single run.

`--improve_*` arguments specify the way we try a new robust radius to certify. The most precise way is to conduct binary search as listed in Algorithm 2 in the paper, but for efficiency we can also use `grid` mode as `improve_mode` which iteratively enlarges the radius by `imrpove_unit` and tries to certify.

As mentioned in Appendix E.3, among these three steps, the most time-consuming step is Step 1 on typical image classification datasets.

##### Result Summarization and Plotting

We provide the script `dump_main_result.py` to summarize main experimental data in our paper.

Usage: `python dump_main_result.py`
It will create `result/` folder and dump all main tables and figures there. Some critical results are also printed in stdout.

##### Appendix: Training Scripts

The repo also contains code that trains the model suitable for DSRS certification (as discussed in Appendix I).

`train.py` code is adapted from Consistency training code in https://github.com/jh-jeong/smoothing-consistency.

`train_smoothmix.py` code is adapted from SmoothMix training code in https://github.com/jh-jeong/smoothmix.

For MNIST and CIFAR-10, we train from scratch. For ImageNet, we finetune from pretrained ImageNet models for a few epochs.
Whent the training is finished, we need to copy the `*.pth.tar` model to `models/` folder.

- Gaussian augmentation training:
  - MNIST: `python train.py mnist mnist_43 --lr 0.01 --lr_step_size 50 --epochs 150  --noise 0.25/0.50/1.00/... --num-noise-vec 1 --lbd 0 --k 380 --k-warmup 100`
  - CIFAR-10: `python train.py cifar10 cifar_resnet110 --lr 0.01 --lr_step_size 50 --epochs 150  --noise 0.25/0.50/1.00/... --num-noise-vec 1 --lbd 0 --k 1530 --k-warmup 100`
  - ImageNet: `python train.py imagenet resnet50 --lr 0.001 --lr_step_size 1 --epochs 6  --noise 0.25/0.50/1.00 --num-noise-vec 1 --lbd 0 --k 75260 --k-warmup 60000 --batch 96 --pretrained-model ../../pretrain_models/cohen_models/models/imagenet/resnet50/noise_[0.25/0.50/1.00]/checkpoint.pth.tar`
    
    Note: the pretrained models are from Cohen et al's randomized smoothing [repo](https://github.com/locuslab/smoothing), and the direct link is here: https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view.

- Consistency training:
  - MNIST: `python train.py mnist mnist_43 --lr 0.01 --lr_step_size 50 --epochs 150  --noise 0.25/0.50/1.00 --num-noise-vec 2 --lbd 5 --k 380 --k-warmup 100`
  - CIFAR-10: `python train.py cifar10 cifar_resnet110 --lr 0.01 --lr_step_size 50 --epochs 150  --noise 0.25/0.50/1.00 --num-noise-vec 2 --lbd 20 --k 1530 --k-warmup 100`
  - ImageNet: `python train.py imagenet resnet50 --lr 0.001 --lr_step_size 1 --epochs 6  --noise 0.25/0.50/1.00 --num-noise-vec 2 --lbd 5 --k 75260 --k-warmup 60000 --batch 96 --pretrained-model ../../pretrain_models/cohen_models/models/imagenet/resnet50/noise_[0.25/0.50/1.00]/checkpoint.pth.tar`
  
- SmoothMix training:
  - MNIST: `python train_smoothmix.py mnist mnist_43 --lr 0.01 --lr_step_size 30 --epochs 90  --noise_sd 0.25/0.50/1.00 --eta 5.00 --num-noise-vec 4 --num-steps 8 --mix_step 1 --k 380 --k-warmup 70`
  - CIFAR-10: `python train_smoothmix.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise_sd 0.5 --eta 5.00 --num-noise-vec 2 --num-steps 4 --alpha 1.0 --mix_step 1 --k 1530 --k-warmup 110`
  - ImageNet: `python train_smoothmix.py imagenet resnet50 --lr 0.01 --lr_step_size 30 --epochs 10  --noise_sd 0.5 --eta 1.00 --num-noise-vec 1 --num-steps 1 --alpha 8.0 --mix_step 0 --k 75260 --k-warmup 200000 --batch 48 --pretrained-model ../../pretrain_models/cohen_models/models/imagenet/resnet50/noise_0.50/checkpoint.pth.tar`



### Guidelines for Reproducing Experimental Results


Training -> certifying pretrained models -> dump tables/figures from certification data

> Scenario A: Reproducing by dumping tables/figures from certification data? (in seconds)

The original experimental data (sampled pA, qA, and computed robust radius from Neyman-Pearson-based certification and DSRS certification) is in `data/` (uploaded to the repo). We only need to run `python dump_main_result.py` and browse the results in `result/` folder.

> Scenario B: Reproducing from certifying pretrained models? (1-2 days on 24-core CPU)
 
First, download the pretrained models and unzip to `models/` folder.

Then, run the three steps in *Usage > Running Certification from Scratch*.

1. Step 1:

  a. Sampling from P:
  - For MNIST models:
    
    MODEL = models/new_cohen/cohen-mnist-380-[0.25/0.50/1.00].pth.tar (for Gaussian augmentation models)
    
    or MODEL = models/consistency/consistency-mnist-380-[0.25/0.50/1.00].pth.tar (for Consistency models)
    
    or MODEL = models/smoothmix/smoothmix-mnist-380-[0.25/0.50/1.00].pth.tar (for SmoothMix models)

    `python sampler.py mnist MODEL 0.25/0.50/1.00 --disttype general-gaussian --k 380 --N 50000 --alpha 0.0005 --skip 10 --batch 400`

  - For CIFAR-10 models:

    MODEL = models/new_cohen/cohen-cifar-1530-[0.25/0.50/1.00].pth.tar (for Gaussian augmentation models)

    or MODEL = models/consistency/consistency-cifar-1530-[0.25/0.50/1.00].pth.tar (for Consistency models)

    or MODEL = models/smoothmix/smoothmix-cifar-1530-[0.25/0.50/1.00].pth.tar (for SmoothMix models)

    `python sampler.py cifar10 MODEL 0.25/0.50/1.00 --disttype general-gaussian --k 1530 --N 50000 --alpha 0.0005 --skip 10 --batch 400`

  - For ImageNet models:

    MODEL = models/new_cohen/cohen-imagenet-75260-[0.25/0.50/1.00].pth.tar (for Gaussian augmentation models)

    or MODEL = models/consistency/consistencyn-imagenet-75260-[0.25/0.50/1.00].pth.tar (for Consistency models)

    or MODEL = models/smoothmix/smoothmix-imagenet-75260-[0.25/0.50/1.00].pth.tar (for SmoothMix models)

    `python sampler.py imagenet MODEL 0.25/0.50/1.00 --disttype general-gaussian --k 75260 --N 50000 --alpha 0.0005 --skip 50 --batch 400`

  b. Sampling from Q:

  - For MNIST models:

    `python sampler.py mnist MODEL 0.2/0.4/0.8 --disttype general-gaussian --k 380 --N 50000 --alpha 0.0005 --skip 10 --batch 400`

  - For CIFAR-10 models:

    `python sampler.py cifar10 MODEL 0.2/0.4/0.8 --disttype general-gaussian --k 1530 --N 50000 --alpha 0.0005 --skip 10 --batch 400`

  - For ImageNet models:

    `python sampler.py imagenet MODEL 0.25/0.50/1.00 --disttype general-gaussian --k 1530 --N 50000 --alpha 0.0005 --skip 50 --batch 400 --th x+`

  c. (optional) Sampling from P for the baseline: replicating the commands in Step 1a but change arguments to ``--N 100000 --alpha 0.001``, since Neyman-Pearson-based certification only needs one sampling process so we can allocate all 10e5 sampling number and 0.001 confidence budget in one run.

2. Step 2: Neyman-Pearson-based certification

  a. Prepare for DSRS certification via computing initial Neyman-Pearson-based certification
  - For MNIST models:

    MODEL = cohen-mnist-380-[0.25/0.50/1.00].pth (for Gaussian augmentation models)
    
    or MODEL = consistency-mnist-380-[0.25/0.50/1.00].pth (for Consistency models)
    
    or MODEL = smoothmix-mnist-380-[0.25/0.50/1.00].pth (for SmoothMix models)
    
    `python main.py mnist origin MODEL general-gaussian --k 380 0.25/0.50/1.00 50000 0.0005 --workers 20`

  - For CIFAR-10 models:

    MODEL = cohen-cifar-1530-[0.25/0.50/1.00].pth (for Gaussian augmentation models)

    or MODEL = consistency-cifar-1530-[0.25/0.50/1.00].pth (for Consistency models)

    or MODEL = smoothmix-cifar-1530-[0.25/0.50/1.00].pth (for SmoothMix models)

    `python main.py cifar10 origin MODEL general-gaussian --k 1530 0.25/0.50/1.00 50000 0.0005 --workers 20`

  - For ImageNet models:
  
    MODEL = cohen-imagenet-75260-[0.25/0.50/1.00].pth (for Gaussian augmentation models)
  
    or MODEL = imagenet-75260-[0.25/0.50/1.00].pth (for Consistency models)
  
    or MODEL = smoothmix-imagenet-75260-[0.25/0.50/1.00].pth (for SmoothMix models)
  
    `python sampler.py imagenet origin MODEL general-gaussian --k 75260 0.25/0.50/1.00 50000 0.0005 --workers 20`

  b. (optional) Compute the Neyman-Pearson-based certification as the baseline: replicating the commands in Step 2a but change arguments ``50000 0.0005`` to ``100000 0.001``.

3. Step 3: DSRS certification

  Use the same MODEL variable in Step 2a.

  - For MNIST models: 

    `python main.py mnist improved MODEL general-gaussian --k 380 0.25/0.50/1.00 50000 0.0005 -b 0.2/0.4/0.8 --workers 20`

  - For CIFAR-10 models:

    `python main.py cifar10 improved MODEL general-gaussian --k 1530 0.25/0.50/1.00 50000 0.0005 -b 0.2/0.4/0.8 --workers 20`

  - For ImageNet models:

    `python main.py imagenet improved MODEL general-gaussian-th --k 75260 0.25/0.50/1.00 50000 0.0005 -b x+ --workers 20`

4. Dump the results following the commands in *Scenario A*.


> Scenario C: Reproducing from training, i.e., reproducing from scratch? (1-2 weeks for training on 4x3090 GPU + 1-2 days on 24-score CPU)

First, run training commands: see *Usage > Appendix: Training Scripts*.

Then, run certification commands: see *Scenario B*.

Lastly, dump the results: see *Scenario A*.


### File Organization

Inference scripts:

- sampler.py: obtain pA by sampling predictions for noised inputs

- benign_sampler.py: obtain the smoothed models' benign accuracy via sampling

- main.py: script for computing Neyman-Pearson-based certification and DSRS certification

Core algorithm implementation:

- algo/algo.py

- utils: utility functions for core algorithm

Framework modules:

- architectures.py & archs/: define model architectures

- distributions.py: define distribution sampling functions and implement some Neyman-Pearson-based certification

- smooth.py: smoothed classifier wrapper

- th_heuristic.py: heuristics for Q distribution parameter determination

Training scripts:

- train.py: Gaussian augmentation training, Consistency training

- train_smoothmix.py: SmoothMix training

- train_utils & third_party/: helper module for training

Result analysis:

- dump_main_result.py: dump the main results in the paper

- ablation/: code for ablation studies and theoretical analysis

Data files:

- data/: all main result data

- some of ablation/ files: result data for abaltion study and theoretical analysis

- figures/: all ablation study figures

- result/: condensed data for table typsetting and curves shown in the paper.

### Reference

If you find this repo useful, please read our paper to know all technical details, and consider citing our work:

```bibtex
@InProceedings{li2022double,
  title = 	 {Double Sampling Randomized Smoothing},
  author =       {Li, Linyi and Zhang, Jiawei and Xie, Tao and Li, Bo},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {13163--13208},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
}
```

`
Linyi Li, Jiawei Zhang, Tao Xie, Bo Li. Double sampling Randomized smoothing. Proceedings of the 39th International Conference on Machine Learning, PMLR 162:13163-13208, 2022.
`

### Version History

Initial release (v1.0): Oct 11

Add attack component - attack.py, dump_attack_results.py (v1.1): Feb 7

Add original data/ folder for reproducibility (v1.2): May 16

Current maintainer: Linyi Li (@llylly, linyi2@illinois.edu)

### License

Our library is released under the BSD 3-Clause license.
