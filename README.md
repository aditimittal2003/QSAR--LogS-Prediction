# Molecular Solubility Predictor 🧪

A simple web application that predicts the aqueous solubility (LogS) of a molecule from its SMILES string. This project uses a QSAR model trained on the [Delaney (ESOL) dataset](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv), wrapped in a Flask API, and served to a clean, user-friendly frontend.

## ## Demo

<p align="center">
  <strong>Application Demo</strong>
</p>

<table align="center">
  <tr>
    <td align="center">Initial View</td>
    <td align="center">Successful Prediction (Inside AD)</td>
    <td align="center">Prediction with AD Warning</td>
  </tr>
  <tr>
    <td>
      <img src="assets/app-initial-view.png" alt="Initial view of the predictor" width="100%">
    </td>
    <td>
      <img src="assets/app-prediction-success.png" alt="Successful prediction for Glycerol" width="100%">
    </td>
    <td>
      <img src="assets/app-ad-warning.png" alt="Prediction for Hexaiodobenzene with an AD warning" width="100%">
    </td>
  </tr>
</table>

---
## ## Features

* **🧪 LogS Prediction:** Predicts the solubility of a molecule.
* **⚛️ 2D Molecule Visualization:** Displays the 2D structure of the input molecule.
* **🚦 Applicability Domain (AD) Check:** Warns the user if the input molecule is too different from the model's training data, ensuring more reliable predictions.

---
## ## Quick Start

Follow these steps to get the application running on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

### 2. Set Up the Environment

This will create a virtual environment and install all the necessary packages.

```bash
# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application
The application requires a two-step launch: training the model (a one-time step) and then running the server.

**Step 1: Train the Model**
This script will first automatically download the required dataset and then train the model. The necessary files will be saved to the `saved_model/` directory.

```bash
python train_and_save_model.py
```

**Step 2: Start the Backend Server**
Now, run the Flask server. Keep this terminal window open.

```bash
python app.py
```

**Step 3: Launch the Frontend**

Finally, open the `index.html` file in your web browser to use the application.

---
