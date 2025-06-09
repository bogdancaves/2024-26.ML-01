from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def hello():
    data = request.get_json()
    riga = data["data"]

    # Convert the input dictionary to a pandas DataFrame
    riga_df = pd.DataFrame([riga])

    # Load model
    mymodel = joblib.load("artifact.joblib")
    result = mymodel.predict(riga_df)

    response = {
        'result': {
            'value': result.tolist()  # Convert numpy array to list for JSON serialization
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)