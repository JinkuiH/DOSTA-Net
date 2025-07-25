
---

# DOSTA-Net: Domain-Shuffle Temporal Attention Network for Vessel Extraction in X-Ray Coronary Angiography Using Synthetic Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Artery extraction from X-ray coronary angiography (XCA) images is essential for accurate diagnosis and treatment of coronary artery diseases. This project introduces **DOSTA-Net**, a deep learning framework that leverages synthetic temporal XCA data for training without requiring manual annotations.

📌 **[Paper Link (Coming Soon)]()**
📦 **[Pretrained Model Weights](https://drive.google.com/file/d/1ORcWla7-Ca-b07PasN7dhPU-PVGjxXwF/view?usp=sharing)**


---

## 📁 Project Structure

```
.
├── config.json          # Configuration file for training/inference
├── dataset.py           # Dataset loading and preprocessing
├── inference.py         # Inference script using pretrained model
├── training_ours.py     # Training pipeline
└── README.md            # Project documentation
```

---

## 🟠 Installation

```bash
git clone https://github.com/JinkuiH/DOSTA-Net.git
cd DOSTA-Net
conda create -n dostanet python=3.9
conda activate dostanet
pip install -r requirements.txt
```

---

## 🟠 Inference

Before running inference, please download the pretrained model weights and place them in the weights/ folder. 

You can download our pretrained model from the [Releases](https://drive.google.com/file/d/1ORcWla7-Ca-b07PasN7dhPU-PVGjxXwF/view?usp=sharing) page.

To run inference using pretrained weights:

```bash
python inference.py
```

Results will be saved in the directory specified in the outputs file.

---

## 🟠 Training

To train the model using synthetic and pseudo-labeled data:

```bash
python training_ours.py
```

Make sure `config.json` is properly set (dataset paths, hyperparameters, etc.).

---


## 🟠 Citation

If you find this work helpful, please cite:

```bibtex
@article{hao2025dosta,
  title={DOSTA-Net: Domain-Shuffle Temporal Attention Network for Vessel Extraction in X-Ray Coronary Angiography Using Synthetic Data},
  author={Hao, Jinkui and others},
  journal={TBD},
  year={2025}
}
```



