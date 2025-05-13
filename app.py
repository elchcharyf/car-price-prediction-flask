from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# === CHARGEMENT DES FICHIERS ===
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Fichier 'best_model.pkl' introuvable.")

try:
    with open('scaler_dt.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("Fichier 'scaler_dt.pkl' introuvable.")

try:
    with open('features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    print("Fichier 'features.pkl' introuvable.")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Récupérer les données de l'utilisateur
        symboling = int(request.form['symboling'])
        fueltype = request.form['fueltype']
        aspiration = request.form['aspiration']
        doornumber = request.form['doornumber']
        carbody = request.form['carbody']
        drivewheel = request.form['drivewheel']
        enginelocation = request.form['enginelocation']
        wheelbase = float(request.form['wheelbase'])
        enginesize = float(request.form['enginesize'])
        fuelsystem = request.form['fuelsystem']
        boreratio = float(request.form['boreratio'])
        stroke = float(request.form['stroke'])
        compressionratio = float(request.form['compressionratio'])
        horsepower = int(request.form['horsepower'])
        peakrpm = int(request.form['peakrpm'])
        citympg = int(request.form['citympg'])
        highwaympg = int(request.form['highwaympg'])

        # Créer un DataFrame avec les données de l'utilisateur
        user_input = pd.DataFrame([[symboling, fueltype, aspiration, doornumber, carbody,
                                    drivewheel, enginelocation, wheelbase, enginesize,
                                    fuelsystem, boreratio, stroke, compressionratio,
                                    horsepower, peakrpm, citympg, highwaympg]],
                                  columns=['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody',
                                           'drivewheel', 'enginelocation', 'wheelbase', 'enginesize',
                                           'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
                                           'horsepower', 'peakrpm', 'citympg', 'highwaympg'])

        user_input = pd.get_dummies(user_input)
        user_input = user_input.reindex(columns=feature_names, fill_value=0)
        user_input_scaled = scaler.transform(user_input)

        predicted_price = model.predict(user_input_scaled)
        predicted_price = predicted_price[0]

        return render_template('index.html', price=predicted_price)

    return render_template('index.html', price = None)


if __name__ == '__main__':
    app.run(debug=True)
