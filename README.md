# 🎫 Ticket Classification ML App

This is a Streamlit-based web app that classifies customer complaint tickets into appropriate departments (like HR, IT, or Transportation) using a Support Vector Machine (SVM) classifier trained on text embeddings.

## 💡 Features

- Upload ticket data in CSV format
- Generate sentence embeddings using `sentence-transformers`
- Train and evaluate an SVM classification model
- Predict the department of new complaints
- Save the trained model for later use
- View pending tickets department-wise

## 🧠 Tech Stack

- Python
- Streamlit
- Scikit-learn (SVM)
- Sentence-Transformers
- Pinecone (optional for advanced vector store usage)
- Pandas
- Joblib


## 🗂 Folder Structure

```

.
├── documents/                      # Input PDF complaint documents
│   └── example.pdf
│
├── pages/                          # All Streamlit logic & ML backend
│   ├── admin_backend.py            # All helper/model functions
│   ├── ML_classification.py        # UI for training, evaluation, and prediction
│   ├── load_data.py                # (Optional) CSV loading/preview utility
│   └── pending_tickets.py          # Displays tickets grouped by department
│
├── user_backend.py                
├── app.py
├── modelsvm.pk1 
├── requirement.txt                # List of dependencies
└── README.md                      # This beautiful documentation
````

## 🧾 How to Run

1. Clone the repo or download the files.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt


3. Run Streamlit:

   ```bash
   streamlit run app.py
   ```

> 📌 Make sure to set your Pinecone API key in a `.env` file if you're using Pinecone:

```
PINECONE_API_KEY=your_key_here
```
> 💌 Sample CSV Format
```
Text,Label
"Rude driver with scary driving",Transportation
"Laptop not working",IT
"Need leave approval",HR
```


