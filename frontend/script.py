from flask import Flask, render_template, request
import requests
from cases_model import c_model, d_model

app = Flask("app")

@app.route("/")
def homepage():
    return render_template("homepage.html")

@app.route("/covid_maps")
def maps_index_page():
    return render_template("map_index_page.html")


@app.route("/covid_ai", methods=["GET", "POST"])
def covid_ai():
    print(request.method)
    if request.method == "GET":
        return render_template("covid_ai.html", Cases="", noCases="", Deaths="", noDeaths="")
    elif request.method == "POST":
        country = request.form["country"]
        date = request.form["forecastDate"]
        c_pred = c_model.gen_prediction(date=date, country=country)
        d_pred = d_model.gen_prediction(date=date, country=country)
        return render_template("covid_ai.html",
                               header=f"Predicted cumulative number of cases and Death for {country} ({date}):",
                               noCases=f"Cumulative Cases: {c_pred}",
                               noDeaths=f"Cumulative Deaths: {d_pred}")
    
if __name__ == '__main__':
    app.run()
