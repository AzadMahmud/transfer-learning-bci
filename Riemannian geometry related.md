## **Riemannian-geometry–related MI/EEG papers**

### Main Riemannian-focused or Riemannian-including works

| Title | Key takeaway (Riemannian-related) | Dataset / Task | Performance (as reported) | DOI / Link* | Citations |
|---|---|---|---|---|---|
| Classification Methods for EEG Patterns of Imaginary Movements | Review comparing CSP, deep learning, and **Riemannian geometry** for MI-EEG; reports offline average accuracy ≈ **90.2 ± 6.6%** for Riemannian methods vs 77.5 ± 5.8% (CSP) and 81.7 ± 4.7% (DL); best online Riemannian accuracy for binary MI ≈ **69.3%**  (Kapralov et al., 2021)| Multiple MI BCI studies (review) | Offline means: CSP 77.5%, DL 81.7%, Riemannian 90.2%; best online Riemannian 69.3% (binary)  (Kapralov et al., 2021)| Not given |  (Kapralov et al., 2021)|
| A review of classification algorithms for EEG-based brain–computer interfaces: a 10 year update | Survey of EEG classifiers; **Riemannian geometry classifiers (RMDM, tangent-space)** reach **state-of-the-art** accuracy on many BCI problems and win several BCI competitions; Riemannian methods often match or outperform CSP+xDAWN while needing less data  (Lotte et al., 2018)| Multiple ERP and MI datasets (review) | Riemannian tangent-space methods “clearly outperformed” other SoA methods; RMDM performs as well as CSP+LDA  (Lotte et al., 2018)| Not given |  (Lotte et al., 2018)|
| Riemannian Geometry-Based EEG Approaches: A Literature Review | Up-to-2024 review of EEG+**Riemannian geometry**, especially combinations with deep learning; covers feature extraction, classification, manifold learning, tangent space, and transfer learning; emphasizes robustness to noise/non‑stationarity and transfer learning potential  (Tibermacine et al., 2024)| 42 BCI papers (MI, ERP, SSVEP; review) | Qualitative comparison; no unified accuracy numbers  (Tibermacine et al., 2024)| arXiv (no DOI in abstract) |  (Tibermacine et al., 2024)|
| Research on EEG analysis algorithms based on Riemannian Manifolds | Proposes RMX: tangent-space projection + Riemannian data augmentation + XGBoost for stroke MI-BCI; handles small-sample EEG well; achieves very high accuracy in stroke rehab context  (Xu et al., 2025)| 11 stroke patients, MI-based BCI, 10-fold CV | Mean accuracy **96.5%** (10-fold CV)  (Xu et al., 2025)| Not given |  (Xu et al., 2025)|
| Is Riemannian Geometry Better than Euclidean in Averaging Covariance Matrices for CSP-based Subject-Independent Classification of Motor Imagery? | Compares Euclidean vs **Riemannian (Fréchet) means** for CSP covariance averaging in subject‑independent MI; using Riemannian means yields statistically significant accuracy improvements on a very large MI dataset (54 subjects, 21,600 trials)  (Kainolda et al., 2021)| 54‑subject left/right-hand MI dataset | “Statistically significant better performance” with Riemannian averaging; exact % not specified  (Kainolda et al., 2021)| Not given |  (Kainolda et al., 2021)|
| Combining detrended cross-correlation analysis with Riemannian geometry-based classification for improved brain-computer interface performance | Replaces standard covariance with **DCCA matrices** as input to Riemannian MDM; DCCA‑Riemannian decoder significantly outperforms vanilla Riemannian and CSP-based approaches offline and works in real time online  (Racz et al., 2024)| Offline: in-house 18‑subject left/right MI; PhysioNet EEG Motor Movement/Imagery v1.0.0; Online: 8 subjects MI-BCI | Offline on PhysioNet: best normalized κ ≈ **0.28–0.29** (DCCA‑Riemannian-MDM, depending on scale) vs lower for standard MDM and Cov‑CSP‑LDA  (Racz et al., 2024)| PhysioNet link given; DOI not in abstract |  (Racz et al., 2024)|
| EEG Classification for MI-BCI using CSP with Averaging Covariance Matrices: An Experimental Study | Experimental comparison of **Riemannian vs Euclidean averaging** of covariance matrices within CSP+SVM frameworks on four MI datasets; Riemannian averaging improves accuracy by about **2 percentage points** for low‑dimensional features, but gains vanish with high feature dimension  (Miah et al., 2019)| 4 public MI datasets; tasks: RH vs foot; RH vs LH  (Miah et al., 2019)| ≈ **+2% accuracy** with Riemannian averaging for small feature dimension  (Miah et al., 2019)| Not given |  (Miah et al., 2019)|
| An Online Data Visualization Feedback Protocol for Motor Imagery-Based BCI Training | Uses **Riemannian Potato** for online artifact rejection and Riemannian manifold visualization for feedback; focuses on training effects, not classifier accuracy  (Duan et al., 2021)| 10 subjects, MI-BCI training over 3 days | Reports improvements in class distinctiveness and feature discriminancy, not explicit accuracies  (Duan et al., 2021)| Not given |  (Duan et al., 2021)|
| The Riemannian spatial pattern method: mapping and clustering movement imagery using Riemannian geometry | Introduces **Riemannian Spatial Pattern (RSP)** to extract spatial patterns from Riemannian classifiers; compares to CSP on ECoG MI (arm/finger movements); RSP yields similar decoding, but better clustering/differentiation of imagined movements  (Larzabal et al., 2021)| Single quadriplegic patient, ECoG MI of arm articulations and fingers | “Similar results” to CSP for mapping; higher differentiation in clustering (no explicit % accuracy)  (Larzabal et al., 2021)|
| Review of Riemannian Distances and Divergences, Applied to SSVEP-based BCI | Reviews Riemannian distances/divergences for covariance-based SSVEP BCIs; shows **Riemannian centers of classes outperform Euclidean** in offline and online classification, with some metrics giving best accuracy–efficiency tradeoff  (Chevallier et al., 2020)| SSVEP dataset (covariance‑based classification) | Riemannian centers consistently better than Euclidean; method-dependent accuracy gains (no single % reported)  (Chevallier et al., 2020)|



\*DOI/links are not provided in these abstracts; they can be retrieved reliably by pasting the exact titles into Google Scholar, PubMed, or IEEE Xplore.

### Summary

Across these works, Riemannian methods are used as full classifiers, as means to average covariance matrices, or as tools for visualization and feedback. Reported gains range from modest (~2% over Euclidean averaging)  (Miah et al., 2019)to very high accuracies in specialized or small‑sample clinical MI datasets (up to 96.5%)  (Xu et al., 2025), and multiple reviews conclude that Riemannian approaches achieve or surpass state‑of‑the‑art BCI performance  (Lotte et al., 2018; Tibermacine et al., 2024; Chevallier et al., 2020).
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Chevallier, S., Kalunga, E., Barthélemy, Q., & Monacelli, É. (2020). Review of Riemannian Distances and Divergences, Applied to SSVEP-based BCI. *Neuroinformatics, 19*, 93 - 106. https://doi.org/10.1007/s12021-020-09473-9
 
Duan, X., Xie, S., Xie, X., Obermayer, K., Cui, Y., & Wang, Z. (2021). An Online Data Visualization Feedback Protocol for Motor Imagery-Based BCI Training. *Frontiers in Human Neuroscience, 15*. https://doi.org/10.3389/fnhum.2021.625983
 
Kainolda, Y., Abibullaev, B., Sameni, R., & Zollanvari, A. (2021). Is Riemannian Geometry Better than Euclidean in Averaging Covariance Matrices for CSP-based Subject-Independent Classification of Motor Imagery?. *2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)*, 910-914. https://doi.org/10.1109/embc46164.2021.9629816
 
Kapralov, N., Nagornova, Z., & Shemyakina, N. (2021). Classification Methods for EEG Patterns of Imaginary Movements. *Intelligenza Artificiale, 20*, 94-132. https://doi.org/10.15622/ia.2021.20.1.4
 
Larzabal, C., Auboiroux, V., Karakas, S., Charvet, G., Benabid, A., Chabardès, S., Costecalde, T., & Bonnet, S. (2021). The Riemannian spatial pattern method: mapping and clustering movement imagery using Riemannian geometry. *Journal of Neural Engineering, 18*. https://doi.org/10.1088/1741-2552/abf291
 
Lotte, F., Bougrain, L., Cichocki, A., Clerc, M., Congedo, M., Rakotomamonjy, A., & Yger, F. (2018). A review of classification algorithms for EEG-based brain–computer interfaces: a 10 year update. *Journal of Neural Engineering, 15*. https://doi.org/10.1088/1741-2552/aab2f2
 
Miah, A., Islam, M., & Molla, M. (2019). EEG Classification for MI-BCI using CSP with Averaging Covariance Matrices: An Experimental Study. *2019 International Conference on Computer, Communication, Chemical, Materials and Electronic Engineering (IC4ME2)*, 1-5. https://doi.org/10.1109/ic4me247184.2019.9036591
 
Racz, F., Kumar, S., Kaposzta, Z., Alawieh, H., Liu, D., Liu, R., Czoch, A., Mukli, P., & Millán, J. (2024). Combining detrended cross-correlation analysis with Riemannian geometry-based classification for improved brain-computer interface performance. *Frontiers in Neuroscience, 18*. https://doi.org/10.3389/fnins.2024.1271831
 
Tibermacine, I., Russo, S., Tibermacine, A., Rabehi, A., Nail, B., Kadri, K., & Napoli, C. (2024). Riemannian Geometry-Based EEG Approaches: A Literature Review. *ArXiv, abs/2407.20250*. https://doi.org/10.48550/arxiv.2407.20250
 
Xu, F., Zhao, P., Zhang, Z., Huang, C., Li, S., Huang, S., Feng, C., Zhang, Y., & Leng, J. (2025). Research on EEG analysis algorithms based on Riemannian Manifolds. *Journal of Physics: Conference Series, 3147*. https://doi.org/10.1088/1742-6596/3147/1/012003
 

