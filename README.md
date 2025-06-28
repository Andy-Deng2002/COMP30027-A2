# Project Setup Guide

## Environment Setup
1. Create conda environment from the provided environment.yml file:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate COMP30027_A2
```

3. Update File Paths
- Open the following notebooks and update the `file_path` variable to your project directory:
  - `notebooks/cnn.ipynb`
  - `notebooks/xgboost_svm.ipynb` 
  - `notebooks/ensemble_model.ipynb`

## Running the Models
Execute the notebooks in the following order:

1. Train CNN and SVM/XGBoost models (can be run in parallel):
   - `cnn.ipynb` (~30 minutes)
   - `xgboost_svm.ipynb` (~30 minutes)

2. Run ensemble model:
   - `ensemble_model.ipynb` (after both base models are trained)

## Note
- Ensure you have sufficient computational resources
- Models will be saved in the `models` directory
- The results on test set will be saved in the `results` directory and named as 'submission_{model_name}.csv'

## Data and Models Setup
The `data/`, `models/`, and `results/` directories are not tracked by Git. You need to download them from the following link and place them in the project's root directory:

[Download Data, Models, and Results from Google Drive](https://drive.google.com/file/d/1I2JZidA8NJzks_IPo5PfXVbDV0BErs2i/view?usp=sharing)

After downloading and unzipping, your project structure should look like this:
```
.
├── data/
├── models/
├── notebooks/
├── results/
├── utils/
├── .gitignore
├── app.py
├── environment.yml
└── README.md
```

## Running the Visualization App
To explore the model predictions and performance visually, you can run the Streamlit web application.

1. **Activate the Conda Environment:**
   If it's not already activated, activate the environment:
   ```bash
   conda activate COMP30027_A2
   ```

2. **Navigate to the Correct Directory:**
   From the project root, change into the `notebooks` directory where the app file is located.
   ```bash
   cd notebooks
   ```

3. **Run the Streamlit App:**
   Execute the following command in your terminal:
   ```bash
   streamlit run app.py
   ```
   This will start a local web server and open the application in your default web browser.