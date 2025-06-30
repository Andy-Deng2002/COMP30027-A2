# GTSRB Traffic Sign Recognition - Model Analysis Dashboard

This project provides an interactive Streamlit dashboard to analyze and compare the performance of three different machine learning models (CNN, XGBoost, SVM) on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

The dashboard allows for in-depth error analysis, visualization of misclassifications, and includes Grad-CAM visual explanations for the CNN model.

**[➡️ View the Project Report Here](https://drive.google.com/file/d/1tUErghXo-4WGR16AUVIiuxl7NoO7wwrZ/view?usp=sharing)**

---

## Instructions

### 1. First-Time Setup (Only needs to be done once)

a. **Clone the Repository:**
   Open your terminal, navigate to the directory where you want to store the project, and run the following command:
   ```bash
   git clone https://github.com/Andy-Deng2002/COMP30027-A2.git <folder-name>
   cd <folder-name>
   ```
   *(Replace  `<folder-name>` with the name you want for the project directory)*

b. **Create the Conda Environment:**
   The `environment.yml` file contains all the necessary dependencies. Create the environment by running:
   ```bash
   conda env create -f environment.yml
   ```
   This will create a new, self-contained Conda environment named `comp30027_a2` with all required libraries installed.

### 2. Running the Application (For a Live Demo)

After the one-time setup is complete, follow these two steps each time you want to run the app.

a. **Activate the Environment:**
   ```bash
   conda activate comp30027_a2
   ```

b. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
   Your web browser should automatically open a new tab with the running application. If not, the terminal will provide a local URL (usually `http://localhost:8501`) that you can open manually.

---

### Project Structure
```
.
├── app.py              # The main Streamlit application script
├── environment.yml     # Conda environment specification
├── utils/              # Utility scripts (e.g., for Sankey plot)
├── notebooks/          # Jupyter notebooks for model training and exploration
```