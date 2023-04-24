# FGBC
Code for the paper entitled: "FGBC: Flexible Graph-based Balanced Classifier for Class-imbalanced Semi-supervised Learning". 

# Dependencies

python3.7

torch 1.5.1 (python3.7 -m pip install torch==1.5.1)
torchvision 0.6.1 (python3,7 -m pip install torchvision==0.6.1)
numpy 1.19,4 (python3.7 -m pip install numpy==1.19.4)
scipy (python3.7 -m pip install scipy)
randAugment (python3.7 -m pip install git+https://github.com/ildoonet/pytorch-randaugment), (if an error occurs, type apt-get install git)
tensorboardX (python3.7 -m pip install tensorboadX)
matplotlib (python3.7 -m pip install matplotlib)
progress (python3.7 -m pip install progress)


# How to run
```
python FGBCremix.py --gpu 0 --label_ratio 10 --num_max 500 --imb_ratio 50 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long --out result
```

# Prepare SmallImageNet127 Dataset
see detail in folder prepare_small_imagenet_127


## Citation

Please cite our paper if it is helpful to your work:

```bibtex
@article{kong4240764fgbc,
  title={Fgbc: Flexible Graph-Based Balanced Classifier for Class-Imbalanced Semi-Supervised Learning},
  author={Kong, Xiangyuan and Wei, Xiang and Wang, Jingjie and Liu, Xiaoyu and Xing, Weiwei and Lu, Wei},
  journal={Available at SSRN 4240764}
}
```

# Acknowledgment
This code is constructed based on Pytorch Implementation of ABC(https://github.com/LeeHyuck/ABC)