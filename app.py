import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import requests
import zipfile
import io
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torchvision import transforms

# --- Auto-downloader for Data, Models, and Results ---

def download_and_unzip(url, target_path="."):
    """Downloads a zip file from a URL and extracts it."""
    st.info("首次运行设置：正在下载所需的数据和模型文件（约 185MB），请稍候...")
    
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        with st.spinner("下载中..."):
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        
        with st.spinner("正在解压文件..."):
            # Create a temporary directory for extraction
            temp_extract_dir = os.path.join(target_path, "temp_extract")
            os.makedirs(temp_extract_dir, exist_ok=True)
            zip_file.extractall(temp_extract_dir)

            # Move contents from the nested folder to the project root
            nested_folder = os.path.join(temp_extract_dir, "data_file")
            for item in os.listdir(nested_folder):
                s = os.path.join(nested_folder, item)
                d = os.path.join(target_path, item)
                if os.path.isdir(s):
                    shutil.move(s, d)
                else:
                    shutil.move(s, d)
            
            # Clean up the temporary directory
            shutil.rmtree(temp_extract_dir)

        st.success("文件下载并设置成功！应用正在加载...")
    else:
        st.error(f"无法下载文件。状态码: {response.status_code}")
        st.stop()

def check_and_setup_files():
    """Checks if data/models/results folders exist, if not, downloads them."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, "data")
    
    if not os.path.exists(data_dir):
        # Google Drive direct download link needs to be formatted correctly
        file_id = "1I2JZidA8NJzks_IPo5PfXVbDV0BErs2i"
        gdrive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        download_and_unzip(gdrive_url, project_root)

# Run the setup check at the beginning of the app
check_and_setup_files()


# --- PATH SETUP AND IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from sankey import create_sankey_plot, classes
from GTRSB_CNN import SimpleCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


st.set_page_config(layout="wide")

# --- MODEL AND DATA LOADING ---
@st.cache_resource
def load_cnn_model():
    """Loads the pre-trained CNN model."""
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'cnn_models', 'cnn_fold1_best.pth'))
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_grad_cam_image(model, image_path, target_layer):
    """Generates a Grad-CAM visualization for a given image."""
    image = Image.open(image_path).convert('RGB')
    
    # Define a transform without data augmentation for Grad-CAM
    eval_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    input_tensor = eval_transform(image).unsqueeze(0)
    
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    # Use the original, un-normalized image for visualization
    rgb_img = np.array(image.resize((64, 64))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return visualization

st.title("Model Performance and Error Analysis on GTSRB")

# --- DATA LOADING ---
@st.cache_data
def load_prediction_data(model_name):
    """Loads the predictions and true labels for the selected model."""
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results', 'sankey_data'))
    
    pred_file_map = {
        "CNN": "cnn_val_pred_labels.npy",
        "XGBoost": "xgb_val_pred_labels.npy",
        "SVM": "svm_val_pred_labels.npy"
    }
    
    pred_path = os.path.join(base_path, pred_file_map[model_name])
    true_path = os.path.join(base_path, "y_train.npy") # As found in the notebook
    
    y_pred = np.load(pred_path)
    y_true = np.load(true_path)
    
    return y_pred, y_true

@st.cache_data
def load_image_paths_and_labels():
    """
    Loads image paths and labels from the validation set.
    Also returns a dictionary mapping class IDs to an example image path.
    """
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    train_metadata = pd.read_csv(os.path.join(file_path, "data", "train", "train_metadata.csv"))
    
    # Recreate the train/holdout split to get the same trainset
    trainset, _ = train_test_split(
        train_metadata,
        test_size=0.2,
        random_state=42,
        stratify=train_metadata['ClassId']
    )
    
    image_paths = trainset['image_path'].values
    labels = trainset['ClassId'].values
    
    # Create a mapping from class ID to a sample image path
    class_example_paths = trainset.groupby('ClassId')['image_path'].first().to_dict()
    
    return image_paths, labels, class_example_paths


# --- SIDEBAR ---
st.sidebar.title("Model Selection")
model_selection = st.sidebar.selectbox(
    "Select a model to view its performance on the validation set:",
    ("CNN", "XGBoost", "SVM")
)

# --- Main Page ---
y_pred, y_true = load_prediction_data(model_selection)
image_paths, _, class_example_paths = load_image_paths_and_labels()

# --- PERFORMANCE METRICS ---
st.header(f"{model_selection} Model - Performance Metrics")
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Macro Precision", f"{precision:.4f}")
col3.metric("Macro Recall", f"{recall:.4f}")
col4.metric("Macro F1-Score", f"{f1:.4f}")

with st.expander("View Detailed Classification Report"):
    report = classification_report(y_true, y_pred, target_names=[name for _, name in sorted(classes.items())], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# --- SANKEY PLOT ---
st.header(f"{model_selection} Model - Predicted vs. True Labels Sankey Plot")
fig = create_sankey_plot(y_pred, y_true, model_selection)
st.pyplot(plt.gcf())
plt.clf()

# --- ERROR ANALYSIS: MISCLASSIFIED IMAGE EXPLORER ---
st.header("Error Analysis - Misclassified Image Explorer")

incorrect_indices = np.where(y_pred != y_true)[0]

if len(incorrect_indices) == 0:
    st.success("🎉 No misclassified images found for this model in the validation set!")
else:
    if 'random_incorrect_index' not in st.session_state or st.session_state.get('model_selection') != model_selection:
        st.session_state.random_incorrect_index = np.random.choice(incorrect_indices)
        st.session_state.model_selection = model_selection


    if st.button("Show Another Misclassified Image"):
        st.session_state.random_incorrect_index = np.random.choice(incorrect_indices)
    
    idx = st.session_state.random_incorrect_index
    
    true_label = y_true[idx]
    pred_label = y_pred[idx]
    misclassified_img_path_part = image_paths[idx]
    correct_example_img_path_part = class_example_paths[true_label]
    predicted_example_img_path_part = class_example_paths[pred_label]

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'train'))
    full_misclassified_path = os.path.join(base_path, misclassified_img_path_part)
    full_correct_example_path = os.path.join(base_path, correct_example_img_path_part)
    full_predicted_example_path = os.path.join(base_path, predicted_example_img_path_part)

    st.error(f"**Model incorrectly predicted:** `{classes[pred_label]}` instead of `{classes[true_label]}`")

    col_misclassified, col_correct_example, col_predicted_example = st.columns(3)

    with col_misclassified:
        st.write("#### Misclassified Image")
        image = Image.open(full_misclassified_path)
        st.image(image, caption=f"True: {classes[true_label]} | Predicted: {classes[pred_label]}", use_container_width=True)
    
    with col_correct_example:
        st.write("#### Example of True Class")
        image = Image.open(full_correct_example_path)
        st.image(image, caption=f"An example of: {classes[true_label]}", use_container_width=True)
        
    with col_predicted_example:
        st.write("#### Example of Predicted Class")
        image = Image.open(full_predicted_example_path)
        st.image(image, caption=f"An example of: {classes[pred_label]}", use_container_width=True)

    # --- GRAD-CAM FOR CNN ---
    if model_selection == "CNN":
        st.write("---")
        st.write("### Grad-CAM Analysis")
        with st.spinner("Generating Grad-CAM..."):
            cnn_model = load_cnn_model()
            target_layer = cnn_model.conv4 # Corrected target layer
            grad_cam_img = get_grad_cam_image(cnn_model, full_misclassified_path, target_layer)
            st.image(grad_cam_img, caption="Grad-CAM: Model Attention on Misclassified Image", use_container_width=True) 