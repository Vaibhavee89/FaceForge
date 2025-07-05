# 🧠 GAN for Face Generation using CelebA Dataset

This project implements a Generative Adversarial Network (GAN) to generate human face images using the **CelebA dataset**. The GAN is trained to learn and generate realistic-looking faces after 50 epochs of training.

![Generated Faces](https://github.com/yourusername/yourrepo/assets/generated_faces_sample.png) 

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
* 🎨 Generated face samples at regular intervals to monitor progress

---

## 📦 Dataset

* **Dataset**: [CelebA – CelebFaces Attributes Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
* **Download**: The dataset is downloaded using the Kaggle API key directly inside the Colab notebook.

```python
# Sample code to download using Kaggle API
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d jessicali9530/celeba-dataset
!unzip celeba-dataset.zip -d ./data
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
| 50    | \~1.20         | \~0.65             |

---

## 🖼️ Results

Below are some sample generated face images after training for 50 epochs:

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

---

## 💡 Future Work

* Improve model stability using Wasserstein GAN (WGAN)
* Train on higher resolution (128x128) faces
* Add GUI for face generation

---

## 🧑‍💻 Author

**Vaibhavee Singh**
AI Enthusiast | Deep Learning Researcher
[GitHub](https://github.com/vaibhavee-singh) • [LinkedIn](https://linkedin.com/in/vaibhavee-singh)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🌟 Acknowledgements

* [Kaggle - CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
* TensorFlow Tutorials on GANs
* Deep Learning Specialization by Andrew Ng

```

---

Let me know if you'd like me to customize the notebook filename, insert actual generated samples, or include links to your GitHub repo.
```
# FaceForge
Creating faces that don't exists
