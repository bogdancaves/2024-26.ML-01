from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def hello():
    data = request.get_json()
    riga = data["data"]

    # Convert the input dictionary to a 2D array (list of lists)
    riga_array = [list(riga.values())]

    # Ensure the order of features matches the model's training data
    # Example: riga_array = [[riga['Date'], riga['Hour'], ...]]

    # load model
    mymodel = joblib.load("artifact.joblib")
    result = mymodel.predict(riga_array)
    response = {
        'result': {
            'value': result 
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)