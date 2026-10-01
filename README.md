# MNIST Digit Classifier From Scratch 🧠

A lightweight, two-layer neural network built entirely from scratch using only Python and NumPy. This project classifies handwritten digits (0-9) from the MNIST dataset without relying on high-level machine learning frameworks like TensorFlow or PyTorch. 

The primary goal of this repository is to demonstrate a concrete, under-the-hood understanding of neural network architecture, forward propagation, calculus-based backpropagation, and gradient descent optimization.

---

## 🏗️ Architecture Diagram

The network follows a standard 2-layer Multi-Layer Perceptron (MLP) architecture. Here is the data flow from raw image pixels to final digit prediction:

```text
                                        Input: 28x28 Image (784 Pixels)
                                               │
                                               ▼
                                        ┌─────────────────────────┐
                                        │      Input Layer        │ X
                                        │      (784 nodes)        │
                                        └──────────┬──────────────┘
                                                   │ W1, b1
                                                   ▼
                                        ┌─────────────────────────┐
                                        │     Hidden Layer        │ Z1 = W1 · X + b1
                                        │      (10 nodes)         │ A1 = ReLU(Z1)
                                        └──────────┬──────────────┘
                                                   │ W2, b2
                                                   ▼
                                        ┌─────────────────────────┐
                                        │     Output Layer        │ Z2 = W2 · A1 + b2
                                        │      (10 nodes)         │ A2 = Softmax(Z2)
                                        └──────────┬──────────────┘
                                                   │
                                                   ▼
                                            Prediction (0-9)

```

## ⚡ Features

* **Pure Math Implementation:** All matrix multiplications, activations, and gradient calculations are hard-coded using NumPy.
* **Custom Activation Functions:** Implements Rectified Linear Unit (ReLU) for the hidden layer and a numerically stable Softmax for the output layer.
* **Dynamic Backpropagation:** Uses the chain rule to derive gradients for weights and biases across all layers.
* **One-Hot Encoding:** Custom implementation to map integer labels to categorical probability distributions.

---

## 🧮 The Mathematics

### Forward Propagation

Data moves forward through the network to generate a prediction:

1. **Hidden Layer:**

$$Z^{[1]} = W^{[1]} \cdot X + b^{[1]}$$


$$A^{[1]} = \max(0, Z^{[1]})$$



*(ReLU Activation)*
2. **Output Layer:**

$$Z^{[2]} = W^{[2]} \cdot A^{[1]} + b^{[2]}$$


$$A^{[2]} = \frac{e^{Z^{[2]}_i}}{\sum e^{Z^{[2]}_j}}$$



*(Softmax Activation)*

### Backpropagation & Parameter Updates

The network learns by calculating the error and propagating it backward using gradients:

1. **Error Calculation:**

$$dZ^{[2]} = A^{[2]} - Y$$


2. **Gradients:**

$$dW^{[2]} = \frac{1}{m} dZ^{[2]} \cdot A^{[1]T}$$


$$dW^{[1]} = \frac{1}{m} dZ^{[1]} \cdot X^T$$


3. **Gradient Descent Update:**

$$W = W - \alpha \cdot dW$$


$$b = b - \alpha \cdot db$$



---

## 🚀 Installation & Usage

1. **Clone the repository:**
```bash
git clone [https://github.com/Sahil-Tamrakar/mnist-from-scratch.git](https://github.com/Sahil-Tamrakar/mnist-from-scratch.git)
cd mnist-from-scratch

```


2. **Download the Dataset:**
Ensure `mnist_train_small.csv` is located in the appropriate directory (e.g., `/content/sample_data/` if using Google Colab, or updated in the script to match your local path).
3. **Run the Model:**
Execute the cells in the Jupyter Notebook to initialize parameters, train the model, and view predictions on individual handwritten digits.

---

## 📊 Results

The model undergoes gradient descent optimization (learning rate = $0.10$) for 500 iterations. Despite its simplicity and lack of deep hidden layers, it successfully optimizes to achieve **~84.6% accuracy** on the training set, successfully recognizing standard variations in human handwriting.

---

## 👨‍💻 Author

**Sahil Tamrakar**

GitHub: [@Sahil-Tamrakar](https://www.google.com/search?q=https://github.com/Sahil-Tamrakar)

```

```
