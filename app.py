from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = "change-this-secret-key-before-deploying"  # TODO: move to env var in production

model = pickle.load(open("model.pkl", "rb"))

# --- Encoding maps -----------------------------------------------------
# These match how the model was trained on heart_cleveland_upload.csv
# (all categorical columns in that dataset are zero-indexed).

condition_map = {
    0: "No Heart Disease",
    1: "Heart Disease Detected",
}

USERNAME = "nk"
PASSWORD = "2003"


# --- Auth helpers --------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username", "")
        pwd = request.form.get("password", "")

        if user == USERNAME and pwd == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/home")
@login_required
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        cp = int(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = int(request.form["fbs"])
        restecg = int(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = int(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = int(request.form["slope"])
        ca = int(request.form["ca"])
        thal = int(request.form["thal"])
    except (KeyError, ValueError):
        flash("Please fill in every field with a valid value.")
        return redirect(url_for("home"))

    final = np.array([[age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak,
                        slope, ca, thal]])

    prediction = int(model.predict(final)[0])

    # Confidence score, when the model supports it, drives the result gauge
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(final)[0]
        confidence = round(float(proba[prediction]) * 100, 1)

    result = condition_map[prediction]
    is_positive = prediction == 1

    return render_template(
        "result.html",
        result=result,
        is_positive=is_positive,
        confidence=confidence,
        inputs={
            "Age": age,
            "Resting BP": f"{trestbps} mm Hg",
            "Cholesterol": f"{chol} mg/dl",
            "Max Heart Rate": f"{thalach} bpm",
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
