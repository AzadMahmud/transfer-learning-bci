## **EEG-MI with ViT, Transformers & Transfer Learning:** Recent high-accuracy works

Below focuses on recent (≈2022–2026) MI-EEG papers using Vision/Transformers or explicit transfer learning, highlighting the strongest reported accuracies.

### 1. Transformer / ViT-style Architectures (subject‑specific & cross‑subject)

| Method / Paper | Model type | Dataset(s) | Best reported performance | Citations |
|----------------|-----------|-----------|---------------------------|-----------|
| **STIT-Net – Spatio Temporal Inception Transformer** [alpha/beta bands]  (S & Selvakumari, 2025)| Wavelet + CNN + Transformer encoder | Physionet MI | **95.70%** (binary, beta band), **93.52%** (binary, alpha); 4‑class up to **82.66%** |  (S & Selvakumari, 2025)|
| **CTNet** – Convolutional Transformer Network  (Zhao et al., 2024)| EEGNet‑like CNN + Transformer encoder | BCI IV‑2a, 2b | Subject‑specific: **88.49%** (2b), 82.52% (2a); cross‑subject: 76.27% (2b), 58.64% (2a) |  (Zhao et al., 2024)|
| **Temporal–Spatial Transformer with ICA**  (Hameed et al., 2024)| Temporal and spatial self‑attention | BCI IV‑2a, 2b | Outperforms state‑of‑the‑art in both subject‑dependent and subject‑independent settings (no exact % in abstract) |  (Hameed et al., 2024)|
| **Local & Global Convolutional Transformer**  (Zhang et al., 2023)| CNN + local & global Transformer + DenseNet | KU MI, BCI IV‑2a | Up to **77.14%** within‑session (KU); cross‑session improvements up to **7.49%** (KU) and **2.12%** (IV‑2a) over SOTA |  (Zhang et al., 2023)|
| **Scalogram sets + Modified ViT (MViT)**  (Balendra et al., 2025)| Wavelet scalograms → ViT variant | BCI IV‑2b | **86.34%** intra‑subject; **76.19%** inter‑subject (best among compared methods) |  (Balendra et al., 2025)|
| **Multiscale Convolutional Transformer** (multi‑modal imagery)  (Ahn et al., 2022)| Spatial–spectral–temporal attention | BCI IV‑2a | 6‑class mental imagery accuracy **0.70** (70%) on IV‑2a |  (Ahn et al., 2022)|
| **EEGEncoder – Transformer + TCN (DSTS block)**  (Liao et al., 2025)| TCN + modified Transformer | BCI IV‑2a | Subject‑dependent: **86.46%**; subject‑independent: **74.48%** |  (Liao et al., 2025)|
| **Three-Branch Temporal-Spatial Convolutional Transformer (EEG‑TBTSCTnet)**  (Chen et al., 2024)| 3‑branch CNN + Transformer encoder | Two MI datasets (not named in abstract) | Reported as outperforming baselines; detailed accuracies not in abstract |  (Chen et al., 2024)|

### 2. Transfer Learning for MI-EEG (cross‑subject / cross‑session / cross‑task)

| Paper | Transfer strategy | Dataset(s) | Performance (avg accuracies) | Citations |
|-------|-------------------|-----------|------------------------------|-----------|
| **Transfer Data Learning Network (TDLNet)**  (Bi & Chu, 2023)| Cross‑subject data grouping + attention | UML6, GRAZ (6‑class upper‑limb MI) | **65%±0.05** (UML6, 6‑class), **63%±0.06** (GRAZ, 6‑class) |  (Bi & Chu, 2023)|
| **Explainable Cross‑Task Adaptive TL**  (Miao et al., 2023)| Pretrain on motor execution → finetune on MI | OpenBMI, GIST (4‑class MI) | **80.00%** (OpenBMI), **72.73%** (GIST), best among several SOTA |  (Miao et al., 2023)|
| **Self‑supervised Contrastive Learning (cross‑subject)**  (Li et al., 2024)| Self‑supervised pretraining + CNN‑attention encoder | BCI IV‑2a, 2b, HGD | **67.32%** (2a), **82.34%** (2b), **81.13%** (HGD), better than existing methods |  (Li et al., 2024)|
| **Adaptive Cross‑Subject TL (CSP+KMM+TrAdaBoost)**  (Feng et al., 2022)| Instance‑based transfer on CSP features | BCI IV public datasets + in‑house | Public datasets avg **89.1%**; in‑house **80.4%** |  (Feng et al., 2022)|
| **Improved Wasserstein Domain Adaptation Network**  (She et al., 2023)| GAN‑style domain adaptation with attention | BCI IV‑2a, 2b | Outperforms multiple SOTA methods (no explicit % in abstract) |  (She et al., 2023)|
| **Dual Selections Knowledge Transfer Learning (DS‑KTL)**  (Luo, 2023)| Riemannian tangent‑space + dual selection | BCI IV‑2a (L/R, F/T), 2b | Cross‑subject accuracies improved vs SOTA; best per‑subject up to **91.67%** in one setting  (Luo, 2023)|
| **SSMT – Semi‑supervised Multi‑Source Transfer**  (Zhang et al., 2024)| Multi‑source TL with dynamic weighting | Two MI datasets | Avg accuracies **83.57%** and **85.09%** |  (Zhang et al., 2024)|
| **ConvoReleNet TL (subject‑independent)**  (Otarbay & Kyzyrkanov, 2026)| Pretrain CNN‑relational net → finetune | BNCI IV‑2a, 2b | Best‑case **87.55%** (2a) and **83.85%** (2b); avg gains +7–9 points vs training from scratch |  (Otarbay & Kyzyrkanov, 2026)|
| **Cross‑Session TL with Attention CNN**  (Xiong et al., 2025)| Attention separable CNN + domain alignment | BCI IV‑2a | 4‑class cross‑session accuracy **73.1%**, κ=0.648 |  (Xiong et al., 2025)|
| **Subject‑Adaptive TL with Resting‑State EEG (ResTL)**  (An et al., 2024)| Adapt with resting‑state rather than task data | 3 public MI benchmarks | Achieves SOTA cross‑subject accuracy on all three benchmarks (numbers not in abstract) |  (An et al., 2024)|

### 3. Transformer Landscape & ViT in MI-EEG

- A 2025 review of transformers in EEG highlights **hybrid CNN–Transformer and Time Series Transformers** as the most successful MI architectures, noting a ViT‑based hybrid that achieved “superior MI decoding performance” and cross‑subject accuracies of **81.33% (BCI IV‑2a)** and **86.23% (IV‑2b)** in one study  (Vafaei & Hosseini, 2025).  
- Data augmentation and transfer learning usually add **~3–7%** accuracy for transformer models in MI  (Vafaei & Hosseini, 2025).  

### Summary

Across recent works, **highest subject‑specific MI accuracies with transformer/ViT‑style models** are around **95–96%** for binary MI on Physionet‑like datasets  (S & Selvakumari, 2025). On standard BCI IV‑2a/2b benchmarks, strong **Transformer/CNN hybrids and transfer‑learning frameworks** reach **≈82–89%** (subject‑specific) and **≈74–84%** (subject‑independent or cross‑subject)  (Zhao et al., 2024; Li et al., 2024; Liao et al., 2025; Otarbay & Kyzyrkanov, 2026). Combining transformers or ViT‑like blocks with **wavelets/scalograms** and **transfer learning / domain adaptation** currently gives the most competitive and robust performance.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
, B., Negi, P., Sharma, N., & Sharma, S. (2025). Scalogram sets based motor imagery EEG classification using modified vision transformer: A comparative study on scalogram sets. *Biomed. Signal Process. Control., 104*, 107640. https://doi.org/10.1016/j.bspc.2025.107640
 
Ahn, H., Lee, D., Jeong, J., & Lee, S. (2022). Multiscale Convolutional Transformer for EEG Classification of Mental Imagery in Different Modalities. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31*, 646-656. https://doi.org/10.1109/tnsre.2022.3229330
 
An, S., Kang, M., Kim, S., Chikontwe, P., Shen, L., & Park, S. (2024). Subject-Adaptive Transfer Learning Using Resting State EEG Signals for Cross-Subject EEG Motor Imagery Classification. *ArXiv, abs/2405.19346*. https://doi.org/10.48550/arxiv.2405.19346
 
Bi, J., & Chu, M. (2023). TDLNet: Transfer Data Learning Network for Cross-Subject Classification Based on Multiclass Upper Limb Motor Imagery EEG. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31*, 3958-3967. https://doi.org/10.1109/tnsre.2023.3323509
 
Chen, W., Luo, Y., & Wang, J. (2024). Three-Branch Temporal-Spatial Convolutional Transformer for Motor Imagery EEG Classification. *IEEE Access, 12*, 79754-79764. https://doi.org/10.1109/access.2024.3405652
 
Feng, J., Li, Y., Jiang, C., Liu, Y., Li, M., & Hu, Q. (2022). Classification of motor imagery electroencephalogram signals by using adaptive cross-subject transfer learning. *Frontiers in Human Neuroscience, 16*. https://doi.org/10.3389/fnhum.2022.1068165
 
Hameed, A., Fourati, R., Ammar, B., Ksibi, A., Alluhaidan, A., Ayed, M., & Khleaf, H. (2024). Temporal-spatial transformer based motor imagery classification for BCI using independent component analysis. *Biomed. Signal Process. Control., 87*, 105359. https://doi.org/10.1016/j.bspc.2023.105359
 
Li, W., Li, H., Sun, X., Kang, H., An, S., Wang, G., & Gao, Z. (2024). Self-supervised contrastive learning for EEG-based cross-subject motor imagery recognition. *Journal of Neural Engineering, 21*. https://doi.org/10.1088/1741-2552/ad3986
 
Liao, W., Liu, H., & Wang, W. (2025). Advancing BCI with a transformer-based model for motor imagery classification. *Scientific Reports, 15*. https://doi.org/10.1038/s41598-025-06364-4
 
Luo, T. (2023). Dual selections based knowledge transfer learning for cross-subject motor imagery EEG classification. *Frontiers in Neuroscience, 17*. https://doi.org/10.3389/fnins.2023.1274320
 
Miao, M., Yang, Z., Zeng, H., Zhang, W., Xu, B., & Hu, W. (2023). Explainable cross-task adaptive transfer learning for motor imagery EEG classification. *Journal of Neural Engineering, 20*. https://doi.org/10.1088/1741-2552/ad0c61
 
Otarbay, Z., & Kyzyrkanov, A. (2026). Transfer learning for subject-independent motor imagery EEG classification using convolutional relational networks. *Frontiers in Neuroscience, 19*. https://doi.org/10.3389/fnins.2025.1691929
 
S, C., & Selvakumari, S. (2025). STIT-Net- A Wavelet based Convolutional Transformer Model for Motor Imagery EEG Signal Classification in the Sensorimotor Bands. *Clinical EEG and Neuroscience, 57*, 88 - 100. https://doi.org/10.1177/15500594241312450
 
She, Q., Chen, T., Fang, F., Zhang, J., Gao, Y., & Zhang, Y. (2023). Improved Domain Adaptation Network Based on Wasserstein Distance for Motor Imagery EEG Classification. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31*, 1137-1148. https://doi.org/10.1109/tnsre.2023.3241846
 
Vafaei, E., & Hosseini, M. (2025). Transformers in EEG Analysis: A Review of Architectures and Applications in Motor Imagery, Seizure, and Emotion Classification. *Sensors (Basel, Switzerland), 25*. https://doi.org/10.3390/s25051293
 
Xiong, C., Li, H., Wang, Y., Lin, Z., Yang, W., Fan, S., & Wang, Y. (2025). Cross-Session EEG Motor Imagery Classification Method Based on Transfer Learning. *2025 IEEE 5th International Conference on Power, Electronics and Computer Applications (ICPECA)*, 342-348. https://doi.org/10.1109/icpeca63937.2025.10928860
 
Zhang, F., Wu, H., & Guo, Y. (2024). Semi-supervised multi-source transfer learning for cross-subject EEG motor imagery classification. *Medical & Biological Engineering & Computing, 62*, 1655 - 1672. https://doi.org/10.1007/s11517-024-03032-z
 
Zhang, J., Li, K., Yang, B., & Han, X. (2023). Local and global convolutional transformer-based motor imagery EEG classification. *Frontiers in Neuroscience, 17*. https://doi.org/10.3389/fnins.2023.1219988
 
Zhao, W., Jiang, X., Zhang, B., Xiao, S., & Weng, S. (2024). CTNet: a convolutional transformer network for EEG-based motor imagery classification. *Scientific Reports, 14*. https://doi.org/10.1038/s41598-024-71118-7
 

