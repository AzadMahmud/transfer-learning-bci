## **Motor Imagery EEG Classification Papers (CSP & Related Methods)**

### Papers and Key Details

| Title | Key takeaway | Dataset(s) | Performance (as reported) | DOI / Link | Citations |
|-------|--------------|------------|----------------------------|------------|-----------|
| Temporally Constrained Sparse Group Spatial Patterns for Motor Imagery BCI | Jointly optimizes frequency bands and time windows (TSGSP), yielding more robust CSP features and higher MI-BCI accuracy. | BCI Comp. III IIIa; BCI Comp. IV IIa, IIb | Avg. accuracy: **88.5%, 83.3%, 84.3%** on three datasets | IEEE Trans. Cybern., 2019. doi not given in abstract |  (Zhang et al., 2019)|
| Temporal Combination Pattern Optimization Based on Feature Selection Method for Motor Imagery BCIs | Decomposes trials into time segments, applies CSP per segment, and uses feature selection (MUIN, LASSO, PCA, SWLDA) to optimize temporal patterns, clearly improving over standard CSP. | 3 BCI competition datasets (incl. BCI IV Dataset I) | LASSO best: **88.58%**, with ~6–11% improvement over CSP | Front Hum Neurosci, 2020. doi not given |  (Jiang et al., 2020)|
| Frequency-Optimized Local Region Common Spatial Pattern Approach for Motor Imagery Classification | Uses “local region” CSP and variance-based selection plus frequency optimization to improve accuracy, especially with small samples. | BCI Comp. III IVa; BCI Comp. IV I, IIb | “Substantially improved” accuracy vs recent MI methods (no exact % given) | IEEE TNSRE, 2019. doi not given |  (Park & Chung, 2019)|
| Filter Bank Regularized Common Spatial Pattern Ensemble for Small Sample Motor Imagery Classification | Filter-bank + regularized CSP + mutual information feature selection + ensemble; strongly boosts performance in small-sample settings. | BCI Comp. III IVa | Mean accuracy gains: **+12.34% vs CSP**, +4.47–11.57% vs several CSP variants | IEEE TNSRE, 2018. doi not given |  (Park et al., 2018)|


| Title | Key takeaway | Dataset(s) | Performance (as reported) | DOI / Link | Citations |
|-------|--------------|------------|----------------------------|------------|-----------|
| Correlation-based channel selection and regularized feature optimization for MI-based BCI | Correlation-based channel selection + RCSP + SVM improves accuracy by removing redundant/noisy channels. | BCI IV 1; BCI III IVa, IIIa | Acc. with CCS+RCSP: **81.6%, 87.4%, 91.9%** vs much lower baselines | Neural Networks, 2019. doi not given |  (Jin et al., 2019)|
| Internal Feature Selection Method of CSP Based on L1-Norm and Dempster–Shafer Theory | Redefines CSP objective and fuses features via Dempster–Shafer theory, improving accuracy with low extra cost. | 2 BCI competition datasets | “Significant increase” in MI-BCI performance (no exact %) | IEEE TNNLS, 2020. doi not given |  (Jin et al., 2020)|
| Learning Common Time-Frequency-Spatial Patterns for Motor Imagery Classification | CTFSP learns sparse CSP features across multiple bands and time windows with SVM voting, improving MI-BCI performance. | BCI III IVa, IIIa; BCI IV 1 | Outperforms several state-of-the-art methods (no exact %) | IEEE TNSRE, 2021. doi not given |  (Miao et al., 2021)|
| Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b | FBCSP optimizes subject-specific bands; best performer in BCI IV 2a/2b with high kappa. | BCI IV 2a, 2b | Mean κ: **0.569 (2a)**, **0.600 (2b)** | Front Neurosci, 2012. doi not given |  (Ang et al., 2012)|

**Figure 2:** Additional CSP variants and benchmark FBCSP results.

### Summary

Most papers use BCI Competition III/IV datasets and show that extending CSP in **time**, **frequency**, **space**, or via **feature selection/regularization** substantially improves MI-EEG classification accuracy over standard CSP. Exact DOIs are not present in the abstracts; they can be obtained by searching the titles in a scholarly database.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Ang, K., Chin, Z., Wang, C., Guan, C., & Zhang, H. (2012). Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b. *Frontiers in Neuroscience, 6*. https://doi.org/10.3389/fnins.2012.00039
 
Jiang, J., Wang, C., Wu, J., Qin, W., Xu, M., & Yin, E. (2020). Temporal Combination Pattern Optimization Based on Feature Selection Method for Motor Imagery BCIs. *Frontiers in Human Neuroscience, 14*. https://doi.org/10.3389/fnhum.2020.00231
 
Jin, J., Miao, Y., Daly, I., Zuo, C., Hu, D., & Cichocki, A. (2019). Correlation-based channel selection and regularized feature optimization for MI-based BCI. *Neural networks : the official journal of the International Neural Network Society, 118*, 262-270. https://doi.org/10.1016/j.neunet.2019.07.008
 
Jin, J., Xiao, R., Daly, I., Miao, Y., Wang, X., & Cichocki, A. (2020). Internal Feature Selection Method of CSP Based on L1-Norm and Dempster–Shafer Theory. *IEEE Transactions on Neural Networks and Learning Systems, 32*, 4814-4825. https://doi.org/10.1109/tnnls.2020.3015505
 
Miao, Y., Jin, J., Daly, I., Zuo, C., Wang, X., Cichocki, A., & Jung, T. (2021). Learning Common Time-Frequency-Spatial Patterns for Motor Imagery Classification. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 29*, 699-707. https://doi.org/10.1109/tnsre.2021.3071140
 
Park, Y., & Chung, W. (2019). Frequency-Optimized Local Region Common Spatial Pattern Approach for Motor Imagery Classification. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 27*, 1378-1388. https://doi.org/10.1109/tnsre.2019.2922713
 
Park, S., Lee, D., & Lee, S. (2018). Filter Bank Regularized Common Spatial Pattern Ensemble for Small Sample Motor Imagery Classification. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 26*, 498-505. https://doi.org/10.1109/tnsre.2017.2757519
 
Zhang, Y., Nam, C., Zhou, G., Jin, J., Wang, X., & Cichocki, A. (2019). Temporally Constrained Sparse Group Spatial Patterns for Motor Imagery BCI. *IEEE Transactions on Cybernetics, 49*, 3322-3332. https://doi.org/10.1109/tcyb.2018.2841847
 

