# 👨‍⚕️ Medical Chatbot – Setup Guide

## 🧰 What You Need

Make sure you're using **Python 3.5 or newer**.

### Required Libraries

You’ll need these installed for everything to run smoothly:

- **Flask** – to build the web server.
- **PyTorch** – for the machine learning model.
- **NLTK** – for natural language processing.
- **NumPy** – for numerical operations.
- **Scikit-learn** – for ML-related tasks.
- **Pandas** – to work with data.
- **matplotlib** – for any visualization needs.

---

## 📦 Installing Everything

Before diving into the chatbot, set up your environment properly. I highly recommend using a **virtual environment** to keep your setup clean and isolated.

### On Windows (which you're currently using)

Open your terminal or PowerShell and run:

```bash
py -3 -m venv venv
venv\Scripts\activate
pip install flask torch nltk numpy==1.19.3 sklearn pandas matplotlib
```

### On Linux/macOS (if you switch environments)

```bash
python3 -m venv venv
source venv/bin/activate
pip install flask torch nltk numpy sklearn pandas matplotlib
```

> ⚠️ The `numpy==1.19.3` is specifically chosen for PyTorch compatibility in some setups. If things work fine with the latest version, you can skip pinning it.

---

## 🔤 NLTK Tokenizer Setup

To make sure your chatbot can understand text, you’ll need to download the Punkt tokenizer:

```python
import nltk
nltk.download('punkt')
```

You only need to do this **once**, and you’re set!

---

## 🚀 Running Your Chatbot

1. Navigate to your chatbot directory in the terminal:

```bash
cd path\to\your\MedicalChatbot
```

2. Run the Flask app:

```bash
python -m flask run
```

You’ll see something like this:

```
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

Open that address in your browser to start chatting with **Meddy** – your AI medical assistant!

--