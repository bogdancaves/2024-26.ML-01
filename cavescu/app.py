from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def hello():
    data = request.get_json()
    riga = data["data"]

    #param1 = data.get('param1')
    # load model
    mymodel = joblib.load("artifact.joblib")
    result = mymodel.predict(riga)
    response = {
        'result': {
            'value': result 
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)