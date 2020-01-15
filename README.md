# Bias Correction of Learned Generative Models using Likelihood-Free Importance Weighting

This repository provides a reference implementation for debiased evaluation of generative models as described in the paper:

> Bias Correction of Learned Generative Models using Likelihood-Free Importance Weighting  
> [Aditya Grover](https://aditya-grover.github.io), Jiaming Song, Alekh Agarwal, Kenneth Tran, Ashish Kapoor, Eric Horvitz, Stefano Ermon.  
> Advances in Neural Information Processing Systems (NeurIPS), 2019.  
> Paper: https://arxiv.org/abs/1906.09531  
> Blog: https://www.microsoft.com/en-us/research/blog/are-all-samples-created-equal-boosting-generative-models-via-importance-weighting/?OCID=msr_blog_genmodels_neurips_tw

## Requirements

The codebase is implemented in Python 3.7. To install the necessary requirements, run the following commands:

```
pip install -r requirements.txt
```

## Datasets

These scripts for downloading the CIFAR10 dataset will be called automatically the first time the `main.py` script is run. By default, the dataset will be downloaded in the `datasets/` folder. 

## Pretrained Models and Samples

Pretrained ckpts for the classifier models are coming soon. Additionally, we also plan to provide 100,000 samples drawn from PixelCNN++ and SNGAN models.

## Examples

Training of binary classifier for estimating importance weights, evaluation of sample quality metrics, and all other options is handled by the `run.py` script. Some examples below:


_Training binary classifier for distinguishing real and generated images from pixelCNN++_

```
python run.py --datasetdir='./datasets/' --sampledir='./samples/pixelcnnpp' --modeldir='./models/pixelcnnpp' --epochs=20 --lr=0.001 --use-mlp --use-feature --test-batch-size=100 --train
```

_Default and debiased evaluation of sample quality metrics (w/ and w/o self normalization) for pixelCNN++ model_


```
python run.py --datasetdir='./datasets/' --sampledir='./samples/pixelcnnpp' --modeldir='./models/pixelcnnpp' --use-mlp --use-feature --test-batch-size=100 --self-norm
```

**Note:** The scores reported in the paper are with self-normalization.


### A few other options

* To experiment with clipping and flattening, use the options `--clip` and `--flatten` respectively.
* To check calibration of binary classifier, add the option `--calibration`
* To perform a bias-variance analysis of the various Monte Carlo estimators (as in Appendix B), use the option `--bias-variance`


## Citing

If you find NeuralSort useful in your research, please consider citing the following paper:

> @inproceedings{   
> grover2019bias,   
> title={Bias Correction of Learned Generative Models using Likelihood-Free Importance Weighting },  
> author={Aditya Grover and Jiaming Song and Alekh Agarwal and Kenneth Tran and Ashish Kapoor and Eric Horvitz and Stefano Ermon},  
> booktitle={Advances in Neural Information Processing Systems },  
> year={2019} 
> }
