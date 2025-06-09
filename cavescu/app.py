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

    # Extract the first element if the result is a single-value list
    result_value = result[0] if len(result) == 1 else result.tolist()

    response = {
        'result': {
            'value': result_value  # Return the single value or the full list
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)