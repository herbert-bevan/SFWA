from flask import Flask, request, render_template
import pickle


flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@flask_app.route("/")
def home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():

    discounted_price = float(request.form['discounted_price'])
    discount_percentage = float(request.form['discount_percentage'])
    rating = float(request.form['rating'])
    rating_count = float(request.form['rating_count'])


    features = [[discounted_price, discount_percentage, rating, rating_count]]
    predicted_price = model.predict(features)[0]


    return render_template(
        "index.html",
        prediction_text={
            "actual_price": f"${discounted_price:.2f}",
            "predicted_price": f"${predicted_price:.2f}",
        }
    )


if __name__ == "__main__":
    flask_app.run(debug=True)