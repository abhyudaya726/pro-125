from flask import Flask,jsonify,request
from ModelView_controller import get_predict

app = Flask(__name__)

@app.route("/predict",methods=["POST"])
def predict_data():
    image = request.files.get("alphabet")
    prediction = get_predict(image)
    return jsonify({
        "prediction":prediction,
    }),200
if __name__ == "__main__":
    app.run(debug=True)