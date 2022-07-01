

from flask import Flask, render_template, request, redirect

import os

from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "C:/Users/longw/FlaskApp/static/images"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG", "JPG", "JPEG"]

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:

        return True
    else:
        return False

@app.route("/", methods=["GET", "POST"])
def upload_image():

    if request.files:

        if True:

            image = request.files["image"]

            if image.filename == "":
                print ("Image must have a filename")
                return redirect(request.url)
            
            if not allowed_image(image.filename):
                print("Image extension invalid")
                return redirect(request.url)

            else:
                filename = secure_filename(image.filename)


                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

            print("Image Saved")
            
            return redirect(request.url)

    return render_template("index.html")


if __name__ == "__main__":

    app.run(debug=True)
