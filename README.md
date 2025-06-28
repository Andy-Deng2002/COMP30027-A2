# Project Setup Guide

**View the final project report: [COMP30027-report.pdf](./COMP30027-report.pdf)**

---

## Environment Setup

This project uses `conda` for local environment management and a `requirements.txt` file for cloud deployment on Streamlit Community Cloud.

1. **Local Setup (using Conda):**
   Create the conda environment from the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```
   Then, activate the environment:
   ```bash
   conda activate COMP30027_A2
   ```

2. **Cloud Deployment (Streamlit Cloud):**
   The `requirements.txt` file is automatically used by Streamlit Community Cloud for deployment. No manual steps are needed if you are deploying from the GitHub repository.

---

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

## Live Demo

You can access the interactive web application for model visualization and error analysis via the following link.

**[🚀 Launch the Streamlit App](YOUR_STREAMLIT_APP_LINK_HERE)**

The application is deployed on Streamlit Community Cloud. Please note that the app may go into hibernation due to inactivity and might take a moment to wake up on the first visit.