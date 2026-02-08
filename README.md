# IBaSRec

Official implementation of the paper:  
**"Inductive Bias-aware Semantic–Behavioral Representation Modeling for Sequential Recommendation"**

---

## 📁 Code Structure

The repository is organized into the following key directories and files to facilitate reproduction of experiments reported in the paper:

### Raw Dataset
The raw dataset is available for download at:  
🔗 [Google Drive](https://drive.google.com/file/d/1laPyAj85bmhlnsSmQ9azAnCgewdRZaci/view?usp=sharing)

### Experiment Scripts

- **`run_cold_start_scripts/`**  
  Scripts for reproducing the *cold-start evaluation* experiments.

- **`run_hp_scripts/`**  
  Scripts for reproducing the *hyper-parameter analysis* experiments.

- **`run_ablation_scripts/`**  
  Scripts for reproducing *ablation studies* to validate the contribution of each component.

- **`run_loo_scripts/`**  
  Scripts for reproducing the *leave-one-out evaluation* experiments.

- **`run_long_tail.py`**  
  Standalone script for reproducing experiments on *long-tail items and users* under the leave-one-out evaluation setting.

### Model Implementation

Core model components are implemented in:

- `train.py` – Main training pipeline for SASRec-based models and baselines  
- `models/backbone_sasrec.py` – SASRec backbone implementation  
- `models/modules.py` – Custom modules and layers  

Similarly, `train_bert.py` and `train_gru.py` provide training pipelines for BERT4Rec and GRU4Rec backbones, respectively.

---

## 🙏 Acknowledgements

Our codebase adopts an architecture inspired by  
[**AlphaFuse: Learn ID Embeddings for Sequential Recommendation in Null Space of Language Embeddings**](https://github.com/Hugo-Chinn/AlphaFuse).  
We sincerely appreciate their outstanding contributions to the community.
