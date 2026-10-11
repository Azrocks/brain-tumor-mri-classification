# Brain Tumor MRI Classification

## Overview

This project uses deep learning to classify brain MRI images into
four categories from the dataset:

- Glioma
- Meningioma
- No Tumor
- Pituitary

It compares a custom Convolutional Neural Network (CNN) with
MobileNetV2 transfer learning and provides a Streamlit web
application for uploading an image and viewing model predictions.

**Disclaimer:** This is an educational machine-learning project,
not a medical diagnostic system. Predictions must not be used
to make medical decisions.

## Models

### 1. Custom CNN
A convolutional neural network built using TensorFlow/Keras to
learn image features and classify MRI images.

### 2. MobileNetV2
A pretrained MobileNetV2 model used for transfer learning.
The feature extractor is frozen during the initial training stage,
with a classification head added for the four dataset categories.

## Dataset

The dataset contains four classes: glioma, meningioma, no_tumor,
and pituitary. Images are organized into training, validation,
and test splits.

The original dataset source and license should be added here
before publishing this repository.

## Technologies

- Python
- TensorFlow / Keras
- NumPy and Pandas
- Matplotlib and scikit-learn
- Streamlit
- Pillow

## Model Evaluation

The following table contains the saved model-comparison results:

| Model       |   Accuracy |   Precision (macro) |   Recall (macro) |   F1 (macro) |
|:------------|-----------:|--------------------:|-----------------:|-------------:|
| Custom CNN  |   0.853659 |            0.84105  |         0.847666 |     0.842752 |
| MobileNetV2 |   0.918699 |            0.918061 |         0.915228 |     0.91584  |

Metrics should be interpreted using the held-out test set.
The model with the best overall result should be selected based
on the actual measured metrics, not assumptions.

## Run the Streamlit Application

1. Download or clone this repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in the terminal and upload a supported
   JPG, JPEG, or PNG image.

Ensure the model file and `class_names.json` are in the same
directory as `app.py`.

## Project Structure

```text
BrainTumorProject/
├── app.py
├── README.md
├── requirements.txt
├── models/
│   ├── custom_cnn_final.keras
│   ├── mobilenetv2_final.keras
│   └── class_names.json
└── outputs/
    └── model_comparison.csv
```

The actual deployment folder may contain only the files needed
to run the Streamlit application.

## Limitations

- Performance depends on the dataset and image distribution.
- Predictions on images from other sources may be unreliable.
- A model score is not the probability that a person has a tumor.
- This project has not been validated for clinical use.

## Future Improvements

- Evaluate performance separately for each class.
- Review the confusion matrix and class-wise recall.
- Test on an independent dataset.
- Improve the interface and deploy the application online.
