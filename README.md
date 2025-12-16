# Utilisation-of-uncertainty-estimation-and-features-for-part-segmentation



This repository contains the code and pipeline developed for our research. We propose a **hybrid learning framework** that combines **uncertainty estimation** (via Monte Carlo Dropout) with **biological features** of plants (e.g., ear count, ear ratio) to improve segmentation performance across varied plant architectures.
We have implemented our methodology with GAM and done a comparative study with GTNet



---

## 🔍 Overview

Part segmentation of 3D point clouds in plant phenotyping is often affected by high intra-class variability. To address this, we introduce a two-fold approach:


**Feature-Weighted Hybrid Loss**  
   - Incorporates plant-specific biological features (like **ear count** or **ear ratio**) as weighting factors in the loss function.
   - Helps balance learning across samples with different structural complexities.

Please cite this paper

@inproceedings{inproceedings,
author = {Doonan, John and Williams, Kevin and Corke, Fiona and Zhang, Huaizhong and Liu, Yonghuai},
year = {2025},
month = {01},
pages = {632-641},
title = {Uncertainty and Feature-Based Weighted Loss for 3D Wheat Part Segmentation},
doi = {10.5220/0013312300003912}
}
