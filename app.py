import pickle
from flask import Flask, render_template, request
import numpy
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        final_features = [numpy.array(features)]
        model_path = os.path.join('models', 'modelDT.sav')
        model = pickle.load(open(model_path, 'rb'))
        res = model.predict(final_features)
        if res == 1:
            outcome = 'Air Layak Konsumsi'
        else:
            outcome = 'Air Tidak Layak Konsumsi'
        return render_template('index.html', result=outcome)
    return render_template('index.html', result='Something went wrong')

# run application
if __name__ == "__main__":
    app.run(debug=True)
