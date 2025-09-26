# 🤖 Customer Support Ticket Auto-Triage

An end-to-end machine learning project to automatically classify and route customer support tickets using **Natural Language Processing** and a **RESTful API**.

-----

## 📚 Table of Contents

  * [Project Objective](https://www.google.com/search?q=%23-project-objective)
  * [Project Structure](https://www.google.com/search?q=%23-project-structure)
  * [Tech Stack](https://www.google.com/search?q=%23-tech-stack)
  * [Dataset](https://www.google.com/search?q=%23-dataset)
  * [Workflow & How It Works](https://www.google.com/search?q=%23-workflow--how-it-works)
  * [Setup & Installation](https://www.google.com/search?q=%23-setup--installation)
  * [How to Run & Test](https://www.google.com/search?q=%23-how-to-run--test)
  * [Evaluation Framework](https://www.google.com/search?q=%23-evaluation-framework)
  * [Conclusion](https://www.google.com/search?q=%23-conclusion)

-----

## 🎯 Project Objective

Customer support teams handle thousands of tickets daily. Manually reading and routing each ticket is slow, error-prone, and costly.

👉 This project automates the classification and triage of support tickets into predefined categories, enabling **faster response times**, **reduced manual effort**, and **better customer satisfaction**.

#### Supported Categories:

  * 🐞 **Bug Report**
  * ✨ **Feature Request**
  * ⚙️ **Technical Issue**
  * 💳 **Billing Inquiry**
  * 👤 **Account Management**

-----

## 📂 Project Structure

```
Customer-Support-Ticket-Auto-Triage/
│
├── train.py                     # Training script
├── api.py                       # REST API (FastAPI)
├── test_api.py                  # Testing script
├── tickets.csv                  # Dataset (synthetic)
├── ticket_classifier_model.joblib # Trained ML model
├── label_encoder.joblib         # Stores category label mapping
├── requirements.txt             # Dependencies
├── README.md                    # This documentation
├── .gitignore                   # Files to be ignored by Git
└── venv/                        # Virtual environment (not pushed to repo)
```

-----

## ⚙️ Tech Stack

  * **Language**: **Python 3.8+**
  * **Libraries**:
      * **ML & Data**: **Scikit-learn**, **Pandas**, **NLTK**, **Joblib**
      * **API**: **FastAPI**, **Uvicorn**
      * **Testing**: **Requests**
  * **ML Approach**:
      * Text preprocessing (lowercasing, regex cleaning)
      * **TF-IDF** vectorization (unigrams + bigrams)
      * **Logistic Regression** (multiclass classification)

-----

## 📊 Dataset

The dataset (`tickets.csv`) contains customer support tickets with the following fields:

| Column      | Description                              |
|-------------|------------------------------------------|
| Ticket ID   | Unique identifier for each ticket        |
| Subject     | Short summary of the issue               |
| Description | Detailed explanation of the problem      |
| Category    | **Target variable** (ticket type)        |
| Priority    | Ticket urgency (Low, Medium, High)       |
| Timestamp   | Date/time of ticket creation             |

### Synthetic Dataset

Since a real-world dataset was not provided, a synthetic dataset of **500 rows** was generated.

  * **100 tickets per category** to ensure a balanced class distribution.
  * Each ticket has a realistic subject, description, priority, and timestamp.
  * This balanced approach helps the model become robust and avoid bias towards any single category.

-----

## 🔄 Workflow & How It Works

The project workflow is divided into three main stages:

### 1\. Training (`train.py`)

  * Loads the `tickets.csv` dataset.
  * **Preprocesses text**:
      * Converts text to lowercase.
      * Removes non-alphanumeric characters.
      * Combines `Subject` and `Description` into a single text field.
  * Encodes the `Category` labels into numerical format using `LabelEncoder`.
  * Splits the data into training and testing sets (stratified to maintain category balance).
  * Trains a Scikit-learn **Pipeline** that chains two steps:
    1.  **TF-IDF Vectorizer** (with a max of 2000 features and both unigrams & bigrams).
    2.  **Logistic Regression** Classifier (configured for multiclass problems with balanced class weights).
  * Evaluates the model's performance on the test set.
  * **Saves the trained model** (`ticket_classifier_model.joblib`) and the label encoder (`label_encoder.joblib`) to disk.

### 2\. Serving API (`api.py`)

  * Loads the saved `ticket_classifier_model.joblib` and `label_encoder.joblib`.
  * Starts a **FastAPI** server with a `/predict` endpoint that accepts `POST` requests.
  * The endpoint expects a JSON input with the ticket's subject and description:
    ```json
    {
      "subject": "Payment issue",
      "description": "Charged twice this month on my card."
    }
    ```
  * It preprocesses the input text, predicts the category, and returns the result:
    ```json
    {
      "predicted_category": "Billing Inquiry"
    }
    ```

### 3\. Testing (`test_api.py`)

  * A simple script that sends sample requests to the running API to validate its functionality.
  * It tests both a single ticket and a batch of tickets to confirm the predictions are correct.

-----

## 🚀 Setup & Installation

Follow these steps to set up and run the project locally.

### 1\. Clone Repo & Create Virtual Environment

```bash
git clone <your-repo-url>
cd Customer-Support-Ticket-Auto-Triage
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 2\. Install Dependencies

Install all the required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

-----

## 🛠️ How to Run & Test

### 1\. Train the Model

Run the training script. This will process the data, train the classifier, and save the model files.

```bash
python train.py
```

  * **Expected Output**: You will see the model's accuracy score and a detailed classification report printed in the terminal. The files `ticket_classifier_model.joblib` and `label_encoder.joblib` will be created in your project directory.

### 2\. Run the API

Start the FastAPI server using Uvicorn. The `--reload` flag automatically restarts the server when you make code changes.

```bash
uvicorn api:app --reload
```

  * **Expected Output**: The server will start on `http://127.0.0.1:8000`. You can navigate to `http://127.0.0.1:8000/docs` in your browser to access the interactive **Swagger UI** for API testing.

### 3\. Test the API

With the API running, open a **new terminal** and run the test script.

```bash
python test_api.py
```

  * **Expected Output**: The script will print the predictions for several sample tickets, confirming that the entire pipeline is working correctly.
    ```text
    Single Test Response: {'predicted_category': 'Billing Inquiry'}

    --- Batch Tests ---
    Input: App crashes → Predicted: {'predicted_category': 'Bug Report'}
    Input: Add dark mode → Predicted: {'predicted_category': 'Feature Request'}
    Input: Server down → Predicted: {'predicted_category': 'Technical Issue'}
    Input: Billing problem → Predicted: {'predicted_category': 'Billing Inquiry'}
    Input: Password reset → Predicted: {'predicted_category': 'Account Management'}
    ```

-----

## 📈 Evaluation Framework

As per the assignment PDF, the model is evaluated on a combination of metrics to ensure a balanced assessment of its performance. These metrics are all calculated and printed during the training stage (`train.py`).

  * **Accuracy (40%)**: The overall percentage of correctly classified tickets.
  * **Precision & Recall (30%)**: Measures for minimizing false positives/negatives for each category.
  * **F1-Score (20%)**: The harmonic mean of precision and recall, providing a single balanced score.
  * **Latency (10%)**: The time taken for real-time classification, ensuring quick response times.

-----

## 📌 Conclusion

This project successfully delivers a complete and functional **Customer Support Ticket Auto-Triage System**. The final submission includes:

  * A machine learning model trained to classify support tickets with high accuracy.
  * A robust REST API for serving real-time predictions.
  * A validation script to test the entire system.
  * Comprehensive documentation covering the project's setup, workflow, and evaluation.