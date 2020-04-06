from flask import Flask, render_template, request
import requests

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
        return render_template("covid_ai.html")
    elif request.method == "POST":
        country = request.form["country"]
        date = request.form["forecastDate"]
        print(country)
        print(date)
        return render_template("covid_ai.html")


if __name__ == '__main__':
    app.run()
