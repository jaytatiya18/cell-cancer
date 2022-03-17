from flask import Flask, request, render_template
import pickle as pk
import numpy as np

# Flask constructor
app = Flask(__name__)


# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods=["GET", "POST"])
def gfg():
    if request.method == "POST":
        # getting input with name = fname in HTML form
        radius_mean = float(request.form.get("radius_mean"))
        # getting input with name = lname in HTML form
        texture_mean = float(request.form.get("texture_mean"))
        smoothness_mean = float(request.form.get("smoothness_mean"))
        compactness_mean = float(request.form.get("compactness_mean"))
        symmetry_mean = float(request.form.get("symmetry_mean"))
        fractal_dimension_mean = float(request.form.get("fractal_dimension_mean"))
        radius_se = float(request.form.get("radius_se"))
        texture_se = float(request.form.get("texture_se"))
        smoothness_se = float(request.form.get("smoothness_se"))
        compactness_se = float(request.form.get("compactness_se"))
        symmetry_se = float(request.form.get("symmetry_se"))
        fractal_dimension_se = float(request.form.get("fractal_dimension_se"))

        filename = r'logistic_model.pkl'
        loaded_model = pk.load(open(filename, 'rb'))

        predictionresult = loaded_model.predict([[radius_mean, texture_mean, smoothness_mean,compactness_mean,
                        symmetry_mean, fractal_dimension_mean,radius_se,
                    texture_se, smoothness_se, compactness_se,symmetry_se, fractal_dimension_se]])

        return "The diagnosis " + str((predictionresult[0]))
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)