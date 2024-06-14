# Face_Verifcation
Siamese Network for Face Verification
This project implements a Siamese Network using VGG16 for face verification. The model is trained to determine if two images represent the same person.

# Directory Structure

```bash
├── lfw-deepfunneled              # Directory containing face images
├── matchpairsDevTrain.csv        # CSV file with matching pairs for training
├── mismatchpairsDevTrain.csv     # CSV file with mismatching pairs for training
├── matchpairsDevTest.csv         # CSV file with matching pairs for testing
├── mismatchpairsDevTest.csv      # CSV file with mismatching pairs for testing
└── face_verification.ipynb       # Python notebook for training and evaluating the model
```


## Requirements

- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- pandas
- numpy
- matplotlib

## Setup and Execution

1. **Install the required libraries**:

    ```bash
    pip install tensorflow keras scikit-learn pandas numpy matplotlib
    ```

2. **Ensure the data is in place**:
   - Place the images in the `lfw-deepfunneled` directory.
   - Ensure the CSV files are present in the working directory.

3. **Run the notebook**:

    ```bash
    jupyter notebook face_verification.ipynb
    ```

## Script Overview

- **Data Loading**: Load and label matching/mismatching pairs from CSV files.
- **Preprocessing**: Load images and preprocess pairs into arrays.
- **Data Augmentation**: Use `ImageDataGenerator` to augment training data.
- **Model Definition**: Define the Siamese Network using VGG16, compute L1 distance between embeddings.
- **Training**: Train the model with early stopping and plot training history.
- **Evaluation**: Evaluate the model on the test set, plot ROC curve, and compute ROC-AUC.





