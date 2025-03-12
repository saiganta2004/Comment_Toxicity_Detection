**Comment Toxicity Detection using Bidirectional LSTM**

**Overview**

This project detects toxic comments using a deep learning model trained on the Kaggle Toxic Comment Classification dataset. It utilizes Bidirectional LSTM for sequence processing and Gradio for a user-friendly web-based interface.

**Features**

--> Detects toxicity in comments
--> Uses Bidirectional LSTM for improved contextual understanding 
--> Gradio-based Web UI for easy interaction
Future Upgrade: Implementation of DistilBERT for faster inference

**Dataset**

--> Source: Kaggle - [Toxic Comment Classification](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)

**Preprocessing:**

-->Tokenization using TensorFlow/Keras
-->Stopword removal, special character cleaning
-->Splitting into training and test sets

**Installation**

To run this project locally, install the required dependencies:

!pip install -r requirements.txt

**Running the Project**

1. Load the Trained Model
If you haven't trained the model yet, run the Jupyter Notebook toxicity_detection.ipynb to train and save model weights.

2. Run the Gradio App
To launch the Gradio web interface, execute the following command in a Python script:

import gradio as gr
from tensorflow.keras.models import load_model

# Load your trained model

_model = load_model("models/model_weights.h5")
def predict_toxicity(text):
    # Process input text and predict toxicity (modify as per your pre-processing pipeline)
    prediction = model.predict([text])
    return "Toxic" if prediction > 0.5 else "Not Toxic"
iface = gr.Interface(fn=predict_toxicity, inputs="text", outputs="label")
iface.launch()_

This will open a local web UI where users can input comments and receive toxicity predictions.

**Deployment**

-->Local Deployment: Run the Gradio script above.
-->Future Enhancement: Deploy using Flask/FastAPI and integrate with a web application.

**Results & Insights**

-->Model trained with Binary Cross-Entropy Loss
-->Evaluation Metrics: Accuracy, Precision, Recall, F1-score

**Future Improvements:**

-->Use DistilBERT for better accuracy and faster inference
-->improve pre-processing with advanced NLP techniques
-->Deploy as a cloud-based web service

Repository Structure

📂 comment-toxicity-detection/
 ├── 📄 README.md  # Project description & usage guide
 ├── 📄 requirements.txt  # List of dependencies
 ├── 📂 notebooks/  
 │   ├── toxicity_detection.ipynb  # Jupyter Notebook with training code
 ├── 📂 models/  
 │   ├── model_weights.h5  # Saved model weights
 ├── 📂 dataset/  # Small dataset (if applicable, else provide a download link)
 ├── 📄 .gitignore  # Excludes unnecessary files

**References & Acknowledgments**

1.Kaggle Toxic Comment Dataset
2.TensorFlow/Keras Documentation
3.AI with Noor (for dataset insights)

**License**

This project is open-source and available under the MIT License.

