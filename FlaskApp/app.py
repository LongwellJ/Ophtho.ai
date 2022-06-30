
from flask import Flask, render_template, request, redirect


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def upload_image():

    if request.method == "Post":

        if request.files:

            image = request.files["image"]

            print(image)
            
            return redirect(request.url)

    return render_template("index.html")


if __name__ == "__main__":

    app.run(debug=True)

Model_json = "model.json"
Model_weights = "model.h5"

