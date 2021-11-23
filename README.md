# <p align=center>`ERRNet [Pattern Recognition @2022]`</p>

- **Title:** Fast Camouflaged Object Detection via Edge-based Reversible Re-calibration Network
- **Authors:** Ge-Peng Ji, Lei Zhu, Mingchen Zhuge, Keren Fu*
- **Paper link:** [Elsevier Website](https://www.sciencedirect.com/science/article/pii/S0031320321005902) and [arXiv](https://arxiv.org/abs/2111.03216).

> Our project is under construction. If you find some bugs, please let me know or directly pull the request in the github. Contact E-mail: gepengai.ji@gmail.com.

# News & Updates

- [2021/09/23] :fire: Please attention to our brandnew project SINetV2([arXiv]([arXiv](http://dpfan.net/wp-content/uploads/ConcealedOD_paper.pdf) & [Github Projecr](https://github.com/GewelsJI/SINet-V2)), which achieves the existing SOTA performance compared with other competitors.
- [2021/09/21] We also retrain our method, which again obtains a new improvement compared with previous performance. Note that the retrained model performs slightly differently from the original performance reported in the paper due to randomness in the training phase. The latest results can be found here: [Google Drive](https://drive.google.com/file/d/1GSS8nF5OoIpR0l17qwVfgXzujY9nNw1a/view?usp=sharing), which contains four test datasets (ie., CHAMELEON, CAMO, COD10K, NC4K).
- [2021/09/15] Upload the inference code.

# Introduction

- **Why?** Camouflaged Object Detection (COD) aims to detect objects with similar patterns (e.g., texture, intensity, colour, etc) to their surroundings, and recently has attracted growing research interest. As camouflaged objects often present very ambiguous boundaries, how to determine object locations as well as their weak boundaries is challenging and also the key to this task. 
- **What?** Inspired by the biological visual perception process when a human observer discovers camouflaged objects, this paper proposes a novel edge-based reversible re-calibration network called ERRNet. Our model is characterized by two innovative designs, namely Selective Edge Aggregation (SEA) and Reversible Re-calibration Unit (RRU), which aim to model the visual perception behaviour and achieve effective edge prior and cross-comparison between potential camouflaged regions and background. More importantly, RRU incorporates diverse priors with more comprehensive information comparing to existing COD models. 
- **How?** Experimental results show that ERRNet outperforms existing cutting-edge baselines on three COD datasets and five medical image segmentation datasets. Especially, compared with the existing top-1 model SINet, ERRNet significantly improves the performance by ∼6% (mean E-measure) with notably high speed (79.3 FPS), showing that ERRNet could be a general and robust solution for the COD task.

<p align="center">
    <img src="./assets/framework.png"/> <br />
    <em> 
    Figure 1: The overall pipeline of the proposed ERRNet that contains three main cooperative components, including Atrous Spatial Pyramid Pooling (ASPP) for initiating global prior, Selective Edge Aggregation (SEA) for generating edge prior, and Reversible Re-calibration Unit (RRU) for modulating and refining the NGES Priors in a cascaded manner.
    </em>
</p>

# Code Usage

- Download the [training](https://drive.google.com/file/d/1bTIb2qo7WXfyLgCn43Pz0ZDQ4XceO9dE/view?usp=sharing) & [testing](https://drive.google.com/file/d/120wKRvwXpqqeEejw60lYsEyZ4SOicR3M/view?usp=sharing) dataset from google drive website.
- Download the snapshot file from [Google Driver](https://drive.google.com/file/d/1z0RFqIEqQegfWyTBKztbvrYo-vTGT7LL/view?usp=sharing) and put it on `./snapshot/ERRNet_Snapshot.pth`.
- Inference: Just run `test.py` script to get the final predictions.

# Prediction Results

Our results reported in our PR journal can be downloaded here: [Google Drive](https://drive.google.com/file/d/10i3g4XPBz76nMfU9wZEsKbvmeurSs1Qm/view?usp=sharing).

<p align="center">
    <img src="./assets/prediction_compare.png"/> <br />
    <em> 
    Figure 2: Visual comparison of camouflaged object detection maps produced by different methods. (a) Input images, (b) GT, which stands for the ground truths, (c) camouflaged object detection maps produced by our method, (d) SINet [1], (e) EGNet [24], (f) HTC [20], (g) CPD [23], and (h) PFANet [60].
    </em>
</p>


# Citation

If you find this project useful, please consider citing:

    @article{ji2022fast,
      title = {Fast Camouflaged Object Detection via Edge-based Reversible Re-calibration Network},
      journal = {Pattern Recognition},
      volume = {123},
      pages = {108414},
      year = {2022}
    }
    
# License

The source code is free for research and education use only. Any commercial usage should get formal permission first.
