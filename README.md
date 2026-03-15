# 🌿 Plant Classification using Deep Learning

A deep learning project for classifying plant images into different plant species using a fine-tuned EfficientNetB0 model.  
The system takes an image of a plant and predicts its class with a confidence score.

This project was built as part of a Computer Science graduation project and demonstrates the use of transfer learning for image classification.

---

# 📌 Features

- Classifies **17 different plant types**
- Built using **TensorFlow / Keras**
- Uses **EfficientNetB0 (Transfer Learning)**
- Custom **data pipeline for loading and preprocessing images**
- Supports **single image prediction**
- Achieved **~96% test accuracy**

---

# 🌱 Plant Classes

The model can classify the following plants:

- Aloevera
- Banana
- Coconut
- Corn
- Cucumber
- Curcuma
- Eggplant
- Guava
- Mango
- Orange
- Paddy
- Peperchili
- Pineapple
- Shallot
- Soybeans
- Sweetpotatoes
- Watermelon

---

# 📂 Dataset Structure

The dataset is organized into three folders:
datasets/
│
├── train/
│ ├── aloevera/
│ ├── banana/
│ ├── ...
│
├── val/
│ ├── aloevera/
│ ├── banana/
│
└── test/
├── aloevera/
├── banana/

### Dataset Statistics

| Split | Images per class | Total images |
|------|------------------|-------------|
| Train | 700 | 11,900 |
| Validation | 200 | 3,400 |
| Test | 100 | 1,700 |
| **Total** | — | **17,000 images** |

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/plant-classification.git
cd plant-classification
```
Create environment (recommended):
conda create -n plant_ai python=3.10
conda activate plant_ai
Install dependencies:
`pip install -r requirements.txt`
### Model Training:
## The model uses:

EfficientNetB0 pretrained on ImageNet

Fine-tuning of top layers

Data augmentation
## The trained model will be saved as:
`plant_classifier_model/`
### Model Architecture
# Backbone: EfficientNetB0
# Input size: 224x224
# Output layer: 17 classes
# Loss: Categorical Crossentropy
# Optimizer: `Adam`
### Results
`Test Accuracy:96%`
### Future Improvements
Deploy the model using FastAPI
Create a web interface for uploading plant images
Convert the model to TensorFlow Lite for mobile applications
Expand dataset with more plant species
### Author:
# Mohammed Bilal Weshah
