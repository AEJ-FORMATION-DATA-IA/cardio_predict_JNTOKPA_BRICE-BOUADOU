from flask import Flask, render_template, url_for, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html', title='Cardiac predict')


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():

    data = {
        "AGE": int(request.form['age']),
        "SEXE": request.form['sexe'],
        "TDT": request.form['tdt'],
        "PAR": int(request.form['par']),
        "CHOLESTEROL": int(request.form['cholesterol']),
        "GAJ": int(request.form['gaj']),
        "ECG": request.form['ecg'],
        "FCMAX": int(request.form['fcmax']),
        "ANGINE": request.form['angine'],
        "DEPRESSION ": float(request.form['depression']),
        "PENTE": request.form['pente'],
    }

    donnee_patient = pd.DataFrame(data, index=[0])
    coeur = pd.read_excel('Coeur.xlsx')

    # Normalisation
    for col in coeur.drop(['CŒUR'], axis=1).select_dtypes(np.number).columns:
        donnee_patient[col] = donnee_patient[col] / coeur[col].max()

    # Recodage
    for col in coeur.drop(['CŒUR'], axis=1).select_dtypes('object').columns:
        donnee_patient[col] = donnee_patient[col].astype('category').cat.codes

    model = pickle.load(open('cardio_predict.pkl', 'rb'))
    pred = model.predict(donnee_patient)

    app.logger.debug(pred)

    return render_template('index.html', pred=pred)


if __name__ == '__main__':
    app.run(debug=True)
