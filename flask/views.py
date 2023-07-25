from flask import (Blueprint, jsonify, redirect, render_template, request,
                   url_for)

views = Blueprint(__name__, "views")

@views.route("/views/")
def home():
    return render_template("index.html", name = "James") # Can pass multiple vals to render template which are passed to {{}} in index.html

@views.route("/profile/")
def profile():
    return render_template("profile.html")


@views.route("/json/")
def get_json():
    return jsonify({'name':'james', 'coolness':10})


@views.route("/data/")
def get_data():
    data = request.json
    return jsonify(data)

@views.route("/go_to_home/")
def go_to_home():
    return redirect(url_for("views.home"))
