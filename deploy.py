import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


model = load_model("model_boala_inima.keras")


scaler = joblib.load("scaler.save")


@app.route('/', methods=['GET'])
def get_home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = dict(request.form)
    # Extragem și convertim inputurile în float
    varsta = float(data['varsta'])
    sex = float(data['sex'])
    tip_durere_piept = float(data['tip_durere_piept'])
    tensiune_repaus = float(data['tensiune_repaus'])
    colesterol = float(data['colesterol'])
    glicemie = float(data['glicemie'])
    ekg_repaus = float(data['ekg_repaus'])
    frecventa_maxima = float(data['frecventa_maxima'])
    angina_efort = float(data['angina_efort'])
    depresie_st = float(data['depresie_st'])
    panta_st = float(data['panta_st'])
    vase_principale_colorate = float(data['vase_principale_colorate'])
    talasemie = float(data['talasemie'])

    # Construim array-ul input
    input_array = np.array([[varsta, sex, tip_durere_piept, tensiune_repaus, colesterol, glicemie,
                             ekg_repaus, frecventa_maxima, angina_efort, depresie_st, panta_st,
                             vase_principale_colorate, talasemie]])

    # Aplicăm scalarea
    input_scaled = scaler.transform(input_array)

    # Obținem probabilitățile
    proba = model.predict(input_scaled)[0]

    # Convertim la procente, rotunjite cu 2 zecimale
    procentaj = np.round(proba * 100, 2)

    mesaj = ""
    procentaj_clasa_sanatos = procentaj[0]
    procentaj_clasa_bolnav = procentaj[1]

    if procentaj_clasa_sanatos > 70:
        mesaj = f"Pacientul nu prezintă semne de boală cardiacă. (Fără boală: {procentaj_clasa_sanatos:.2f}%, Cu boală: {procentaj_clasa_bolnav:.2f}%)"
    else:
        mesaj = f"Pacientul prezintă semne de boală cardiacă. (Fără boală: {procentaj_clasa_sanatos:.2f}%, Cu boală: {procentaj_clasa_bolnav:.2f}%)"

    return render_template('index.html', **locals())


if __name__ == '__main__':
    app.run(debug=True)
