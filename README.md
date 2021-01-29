# Interpretable Complex-Valued Neural Networks for Privacy Protection

Implementation of the work by Xiang et al. ([ICLR Poster](https://iclr.cc/virtual/poster_S1xFl64tDr.html), [Paper PDF](http://www.openreview.net/pdf?id=S1xFl64tDr)).
Full report will be available soon at [OpenReview](https://api.openreview.net/forum?id=XX26O1HXupp&noteId=rQEKyH2LlHa).

We examine the reproducibility of the quantitative results reported by Xiang et al. Since no publicly available implementation currently exists, we write our own in PyTorch.

![Structure of the complex-valued neural network](assets/complex-CNN-structure.png?raw=true)

## Claims
As the authors do not provide training details in their work, we do not aim to obtain the exact reported metrics. Instead, we focus on the claims that the proposed complex-valued networks are secure against inversion and property inference attacks while maintaining similar performance as the real-value counterparts.

## Requirements
- Python 3.6 or greater.
- Dependencies can be installed by `pip install -r requirements.txt` 

## Training
We include several shell [scripts](scripts/) with examples on how to train the classification models and the various attacker models. 

## Evaluation Notebook
Our results can be reprduced by running the provided [Jupyter Notebook](results.py). The notebook requires the model checkpoints of our trained models, which can be downloaded [here](https://drive.google.com/file/d/1CjgKd9Hys-fA65JX5TNGTT1ZHKXFN68N/view?usp=sharing). The downloaded zip directory needs to be extracted into the root directory in order for the notebook to work properly.

## Results
### Feature inversion attacks
|Original paper   |Our implementation   |
|-----------------|---------------------|
|![](assets/attack-original.png)|![](assets/attack-ours.png)|

### Baseline models – image classification
|Original paper   |Our implementation   |
|-----------------|---------------------|
|![](assets/classification-errors-original.png)|![](assets/classification-errors-ours.png)|

**Original paper**
![](assets/classification-errors-2-original.png)

**Our implementation**
![](assets/classification-errors-2-ours.png)

### Average error of the estimated rotation angle in radian
|Original paper   |Our implementation   |
|-----------------|---------------------|
|![](assets/angle-attack-original.png)|![](assets/angle-attack-ours.png)|

