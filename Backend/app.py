import json
import torch
import nltk
import pickle
import random
from datetime import datetime
import numpy as np
import pandas as pd
from flask_cors import CORS
from flask import Flask, request, jsonify

from nnet import NeuralNet
from nltk_utils import tokenize, stem, sentence_to_indices

random.seed(datetime.now().timestamp())

# Loading multi-label model
device = torch.device('cpu')
FILE = "models/multi_label_model.pth"
model_data = torch.load(FILE, map_location=device)

input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']
vocab_size = model_data['vocab_size']
tags = model_data['tags']
word2idx = model_data['word2idx']

nlp_model = NeuralNet(vocab_size, 64, hidden_size, output_size, input_size).to(device)
nlp_model.load_state_dict(model_data['model_state'])
nlp_model.eval()

# Loading disease info
symptom_severity = pd.read_csv("data/Symptom-severity.csv")
symptom_severity = symptom_severity.map(lambda s: s.lower().strip(" ").replace(" ", "") if type(s) == str else s)

disease_precaution = pd.read_csv("data/symptom_precaution.csv")
disease_precaution['Disease'] = disease_precaution['Disease'].apply(lambda x: x.lower().strip(" "))

diseases_description = pd.read_csv("data/symptom_Description.csv")
diseases_description['Disease'] = diseases_description['Disease'].apply(lambda x: x.lower().strip(" "))

# Loading symptom list and fitted disease prediction model
with open('data/list_of_symptoms.pickle', 'rb') as data_file:
    symptoms_list = pickle.load(data_file)

with open('models/fitted_model_stacked_final.pkl', 'rb') as modelFile:
    prediction_model = pickle.load(modelFile)

with open("models/label_encoder.pkl", "rb") as enc_file:
    label_encoder = pickle.load(enc_file)

user_symptoms = set()
last_suggested_symptom = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

def get_symptoms_multilabel(sentence, word2idx, model, tags, threshold=0.8):
    tokens = tokenize(sentence.lower())
    stemmed = [stem(w) for w in tokens]
    indices = sentence_to_indices(" ".join(stemmed), word2idx, max_len=10)
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor))
        predictions = output[0].numpy()

    results = [tags[i] for i, p in enumerate(predictions) if p > threshold]
    return results

@app.route('/symptom', methods=['GET', 'POST'])
def predict_symptom():
    global last_suggested_symptom
    sentence = request.json['sentence'].strip().lower()

    greetings = {"hi", "hello", "hey", "good morning", "good afternoon"}
    farewells = {"bye", "goodbye", "see you", "see you later"}
    gratitude = {"thanks", "thank you", "appreciate it"}
    non_symptom_phrases = {"wrong", "bad", "what", "why", "huh", "fix", "stop", "nothing", "never mind", "no", "okay", "understood", "alright"}

    if any(greet in sentence for greet in greetings):
        return jsonify("Hello! I'm Meddy, your medical assistant. Let me know how you're feeling.")
    elif any(word in sentence for word in gratitude):
        return jsonify("You're welcome! Feel free to share any symptoms or questions.")
    elif any(bye in sentence for bye in farewells):
        return jsonify("Take care! I'm here anytime you need medical help.")
    elif any(phrase in sentence for phrase in non_symptom_phrases):
        return jsonify("I understand — could you describe how you're feeling instead?")

    if sentence == "done":
        if not user_symptoms:
            response_sentence = "You haven't told me any symptoms yet."
        elif len(user_symptoms) < 3:
            response_sentence = f"You’ve only provided {len(user_symptoms)} symptom(s). Please share at least 3."
        else:
            x_test = [1 if each in user_symptoms else 0 for each in symptoms_list]
            x_test = pd.DataFrame([x_test], columns=symptoms_list)  # Using DataFrame to match feature names

            if hasattr(prediction_model, "predict_proba"):
                probas = prediction_model.predict_proba(x_test)[0]
                top_indices = np.argsort(probas)[-3:][::-1]
                top_diseases = [(label_encoder.inverse_transform([i])[0].lower(), probas[i]) for i in top_indices]
            else:
                predicted = prediction_model.predict(x_test)[0]
                top_diseases = [(label_encoder.inverse_transform([predicted])[0].lower(), 1.0)]

            response_sentence = f"Based on your symptoms ({', '.join(sym.replace('_', ' ') for sym in user_symptoms)}), here are possible conditions:<br><br>"
            sensitive_diseases = {'aids', 'hiv', 'ebola', 'cancer'}
            banners = []
            for disease, prob in top_diseases:
                description_row = diseases_description.loc[diseases_description['Disease'] == disease]
                description = description_row['Description'].values[0] if not description_row.empty else "No description."
                precautions_row = disease_precaution.loc[disease_precaution['Disease'] == disease]
                precautions = ", ".join([precautions_row[f'Precaution_{i}'].values[0] for i in range(1, 5)]) if not precautions_row.empty else "No precautions."
                if disease in sensitive_diseases:
                    response_sentence += f"<b>{disease.title()}</b>: Sensitive info hidden.<br><br>"
                    banners.append("⚠️ Sensitive condition. Consult a healthcare provider.")
                else:
                    response_sentence += f"<b>{disease.title()}</b><br><i>{description}</i><br><b>Precautions:</b> {precautions}<br><br>"

            if banners:
                response_sentence = "<br>".join(banners) + "<br><br>" + response_sentence

            user_symptoms.clear()
            last_suggested_symptom = None
    else:
        detected_symptoms = get_symptoms_multilabel(sentence, word2idx, nlp_model, tags)

        if not detected_symptoms:
            return jsonify("Hmm, I couldn't recognize any symptoms from that. Could you try describing how you're feeling in a different way?")

        normalized_predictions = [s.lower().strip().replace(" ", "_") for s in detected_symptoms]
        new_syms = [s for s in normalized_predictions if s not in user_symptoms]
        user_symptoms.update(new_syms)
        pretty_names = [sym.replace('_', ' ').capitalize() for sym in new_syms]
        response_sentence = f"Got it! I’ve noted: {', '.join(pretty_names)}. Anything else?" if new_syms else "You’ve already mentioned those symptoms. Anything new?"

    return jsonify(response_sentence)
