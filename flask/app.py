# from views import views

# from flask import Flask



# app = Flask(__name__)
# app.register_blueprint(views, url_prefix = "/")

# if __name__ == "__main__":
#     app.run(debug = True,port = 8000)


import connexion

from flask import render_template

app = connexion.App(__name__, specification_dir="/")
app.add_api("swagger.yml")

@app.route("/")
def home():
    return render_template("dataframe.html")

if __name__ == "_main__":
    app.run(host = "0.0.0.0", port =8000, debug = True)
