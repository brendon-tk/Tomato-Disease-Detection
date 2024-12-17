#Tomato Disease Detection with VGG19

Overview

This repository hosts a project aimed at detecting diseases in tomato plants using a VGG19-based convolutional neural network. The model analyzes images of tomato leaves to classify various plant diseases, helping farmers and agricultural experts make informed decisions.

Features

Deep Learning Architecture: Utilizes the VGG19 model for accurate disease detection.

User-Friendly Notebook: A Jupyter Notebook (vgg19_tomato_disease_detection.ipynb) for model training and evaluation.

Customizable Pipeline: Easily adaptable for different datasets and configurations.

Efficient Deployment: Ready for integration into applications or further research.

Project Structure

Tomato-Disease-Detection/
├── main.py                           # Entry point for running the detection script
├── README.md                         # Project documentation
├── vgg19_tomato_disease_detection.ipynb # Colab for model training and testing

Installation

Clone the Repository:

git clone https://github.com/brendon-tk/Tomato-Disease-Detection.git
cd Tomato-Disease-Detection

Install Dependencies:

pip install -r requirements.txt

Run the Notebook:
Open vgg19_tomato_disease_detection.ipynb in Jupyter Notebook or JupyterLab to train and evaluate the model.

Execute the Script:
For a quick detection run, execute main.py.

python main.py

How It Works

Dataset Preprocessing:

The images of tomato leaves are preprocessed and resized for input into the VGG19 model.

Some Visuals:
![WhatsApp Image 2024-12-17 at 14 38 51_787004ee](https://github.com/user-attachments/assets/cb6024e7-86ab-480b-bf27-f9c80c4aebfb)

![WhatsApp Image 2024-12-17 at 14 38 32_6f14ec7d](https://github.com/user-attachments/assets/dfceb8e3-8b4b-45ee-ac93-70e6c3d7a3b8)

Model Training:

The VGG19 architecture is fine-tuned on the dataset to achieve optimal classification accuracy.

Disease Classification:

The trained model predicts the disease class of input images.

Technologies Used

Programming Language: Python

Deep Learning Framework: TensorFlow/Keras

Notebook: Jupyter Notebook

Contributions

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository, implement changes, and create a pull request.

Author

Brendon TK

For any questions or support, contact: brendon@example.com

License

This project is licensed under the MIT License. Feel free to use and adapt it as needed.

Acknowledgments

Thanks to the open-source community for datasets and pre-trained models.

Inspired by advancements in agricultural AI solutions.

