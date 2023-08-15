from flask import Flask, render_template, request, jsonify
import requests
import pickle

app = Flask(__name__)

model_filename = 'classification_gnb.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

gender_mapping = {'F': 0, 'M': 1}
ssc_b_mapping = {'Central': 0, 'Others': 1}
hsc_b_mapping = {'Central': 0, 'Others': 1}
hsc_s_mapping = {'Arts': 0, 'Commerce': 1, 'Science': 2}
degree_t_mapping = {'Comm&Mgmt': 0, 'Others': 1, 'Sci&Tech': 2}
workex_mapping = {'No': 0, 'Yes': 1}
specialisation_mapping = {'Mkt&Fin': 0, 'Mkt&HR': 1}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ssc_p = float(request.form['ssc_p'])
    hsc_p = float(request.form['hsc_p'])
    degree_p = float(request.form['degree_p'])
    etest_p = float(request.form['etest_p'])
    mba_p = float(request.form['mba_p'])
    gender = request.form['gender']
    ssc_b = request.form['ssc_b']
    hsc_b = request.form['hsc_b']
    hsc_s = request.form['hsc_s']
    degree_t = request.form['degree_t']
    workex = request.form['workex']
    specialisation = request.form['specialisation']

    # Convert categorical data to encoded values using mapping dictionaries
    gender_encoded = gender_mapping[gender]
    ssc_b_encoded = ssc_b_mapping[ssc_b]
    hsc_b_encoded = hsc_b_mapping[hsc_b]
    hsc_s_encoded = hsc_s_mapping[hsc_s]
    degree_t_encoded = degree_t_mapping[degree_t]
    workex_encoded = workex_mapping[workex]
    specialisation_encoded = specialisation_mapping[specialisation]

    features = [ssc_p, hsc_p, degree_p, etest_p, mba_p, gender_encoded, ssc_b_encoded, hsc_b_encoded,
                hsc_s_encoded, degree_t_encoded, workex_encoded, specialisation_encoded]

    prediction= model.predict([features])[0]

    prediction = 'Congratulations! You can get a placement.' if prediction == 1 else 'Sorry, You cannot get a placement.'

    response = {'prediction': prediction}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port = 8000)