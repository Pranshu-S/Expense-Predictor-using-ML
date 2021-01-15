import pickle
from flask import Flask, request, jsonify
from model_files.ml_model import predict_amount
import pandas as pd

app = Flask("Amount Prediction")

@app.route('/', methods=['POST'])
def predict(): 
    spending_data = request.get_json()

    data2 = pd.DataFrame(spending_data, columns = ['vendor','date','description','category','Location'])

    print(type(data2))

    with open('./model_files/model.bin', 'rb') as f_in:
        model=pickle.load(f_in)
        f_in.close()
    
    predictions = predict_amount(data2, model)

    response = {
        'amount' : list(predictions)
    }
    return jsonify(response)


# @app.route('/', methods=['GET'])
# def ping():
#     return "Pinging Model Application!!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9697)