# 🧠 GAN for Face Generation using CelebA Dataset

This project implements a Generative Adversarial Network (GAN) to generate human face that don't exists using the **CelebA dataset**.

![Generated Faces](https://github.com/Vaibhavee89/FaceForge/blob/main/GAN.png) 

---

## 📁 Project Structure

```bash
.
├── GAN_CelebA_Faces.ipynb   # Main training notebook using Colab
├── generated_faces/         # Sample outputs after training
├── models/                  # Saved model weights (optional)
└── README.md                # Project documentation
````

---

## 📌 Features

* ✅ Implemented a basic GAN architecture (Generator & Discriminator)
* 📦 Used the **CelebA dataset** downloaded directly via **Kaggle API**
* 🧪 Trained for **50 epochs** using TensorFlow/Keras
* 🎨 Generated face samples after every epoch to monitor progress

---

## 📦 Dataset

* **Dataset**: [CelebA – CelebFaces Attributes Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
* **Download**: The dataset is downloaded using the Kaggle API key directly inside the Colab notebook.

```python
# Sample code to download using Kaggle API
import os
from google.colab import userdata

# Set up Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')

!kaggle datasets download -d jessicali9530/celeba-dataset
```

---

## 🧠 Model Architecture

### Generator:

* Input: Random noise vector (latent space)
* Layers: Dense → BatchNorm → LeakyReLU → Conv2DTranspose
* Output: 64x64x3 RGB image

### Discriminator:

* Input: 64x64x3 RGB image
* Layers: Conv2D → LeakyReLU → Dropout → Flatten → Dense
* Output: Binary classification (real/fake)

---

## 🏋️ Training

* **Epochs**: 50
* **Batch size**: 128
* **Loss Functions**: Binary Crossentropy for both generator and discriminator
* **Optimizer**: Adam (learning rate: 0.0002, β1=0.5)

### Sample Training Output (after 50 epochs):

| Epoch | Generator Loss | Discriminator Loss |
| ----- | -------------- | ------------------ |
| 13    | \~0.96        | \~0.65             |

---

## 🖼️ Results

We were able to train the model for 13 epochs only because of hardware limitations, but you can train it further Below are some sample generated face images after training for 10 epochs:

![Samples](generated_faces/sample_epoch_50.png) <!-- Replace with actual sample image path -->

---

## 🚀 How to Run

1. Open the [Colab Notebook](./GAN_CelebA_Faces.ipynb)
2. Upload your `kaggle.json` (Kaggle API key)
3. Run all cells to:

   * Download the dataset
   * Preprocess images (resize, normalize)
   * Train the GAN
   * Generate new face samples

---

## ✅ Requirements

* Python 3.8+
* TensorFlow 2.x
* NumPy, Matplotlib
* Google Colab (preferred for GPU)

Install dependencies (for local run):

```bash
pip install tensorflow numpy matplotlib kaggle
```


## 🧑‍💻 Author

**Vaibhavee Singh**
AI Enthusiast | Deep Learning Researcher
[GitHub](https://github.com/vaibhavee89) • [LinkedIn](https://www.linkedin.com/in/vaibhavee-singh-1b7996252/)

---

