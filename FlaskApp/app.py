#app/main.py
#Packages
from flask import Flask, render_template, request, redirect, Response, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import os
from werkzeug.utils import secure_filename
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import datetime
import base64
import io

#Meaningless comment

#Global app parameters
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "C:/Users/longw/FlaskApp/static/images"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG", "JPG", "JPEG"]
app.config["Diagnosis"] = "No Image Submitted"
app.config["cnv_prob"] = "0"
app.config["dme_prob"] = "0"
app.config["drusen_prob"] = "0"
app.config["normal_prob"] = "0"


Bootstrap(app)

class NameForm(FlaskForm):
    name = StringField('Which actor is your favorite?', validators=[DataRequired()])
    submit = SubmitField('Submit')


#What images get allowed
def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:

        return True
    else:
        return False
      
    

#Loading the model path
PATH = "better_model.pt"

#Backend functions

#index page code
@app.route("/", methods=["GET", "POST"])
def upload_image():
    
    #return the answer to the html page
    return render_template("index.html")

#disclaimer page code

@app.route("/disclaimer", methods=["GET", "POST"])
def disclaimer():
    
    
    return render_template("disclaimer.html")

#cnv page code

@app.route("/cnv", methods=["GET", "POST"])
def cnv():
    
    
    return render_template("cnv.html")


#dme page code

@app.route("/dme", methods=["GET", "POST"])
def dme():
    
    
    return render_template("dme.html")


#drusen page code

@app.route("/drusen", methods=["GET", "POST"])
def drusen():
    
    
    return render_template("drusen.html")

#normal page code

@app.route("/normal", methods=["GET", "POST"])
def normal():
    
    
    return render_template("normal.html")    


#cite page code

@app.route("/cite", methods=["GET", "POST"])
def citations():

    date = datetime.datetime.now()

    return render_template("cite.html", date_chicago = date.strftime("%B %d, %Y"), date_mla = date.strftime("%d %B %Y"), date_apa = date.strftime("%Y,  %B %d"))


@app.route("/results", methods=["GET", "POST"])
def results():

    if request.files:

        #get the image from the page
        image = request.files["image"]

        #reload if there is no filename         
        if image.filename == "":
            print ("Image must have a filename")
            return redirect(request.url)

        #reload if the image is not accepted   
        if not allowed_image(image.filename):
            print("Image extension invalid")
            return redirect(request.url)

        #save the image if it is good
        else:
            #filename = secure_filename(image.filename)
            #image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

            #set a dummy database for the image
            data = io.BytesIO()
            #prep the image
            img = Image.open(image).convert('RGB')
            #save the image to the dummy database
            img.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
            
            trans1 = transforms.Resize(256)
            resized = trans1(img)
            trans2 = transforms.CenterCrop(224)
            cropped = trans2(resized)
            trans3 = transforms.ToTensor()
            cropped_tensor = trans3(cropped)
            cropped_tensor = cropped_tensor[None, :]

            #evaluate the image using the model
            model = torch.load(PATH)
            model.eval()
            with torch.no_grad():
                
                #newoutput = model(cropped_tensor)
                output = F.softmax(model(cropped_tensor), dim=1)
                app.config["cnv_prob"] = np.round(((output[0][0].numpy()) *100),2)
                app.config["dme_prob"] = np.round(((output[0][1].numpy()) *100),2)
                app.config["drusen_prob"] = np.round(((output[0][2].numpy()) *100),2)
                app.config["normal_prob"] = np.round(((output[0][3].numpy()) *100),2)
                _, pred = torch.max(output, dim=1)

                
            #analysis of the response of the model
            if pred[0] == 0:
                app.config["Diagnosis"] = "This is CNV"
            else:
                if pred[0] == 1:
                    app.config["Diagnosis"] = "This is DME"
                else:
                    if pred[0] == 2:
                        app.config["Diagnosis"] = "This is Drusen"
                    else:
                        if pred[0] == 3:
                            app.config["Diagnosis"] = "This is Normal"
    
    
    return render_template("results.html", x = app.config["Diagnosis"], z = app.config["cnv_prob"], y = app.config["dme_prob"], w = app.config["drusen_prob"], v = app.config["normal_prob"], img_data=encoded_img_data.decode('utf-8'))



#debugger
if __name__ == "__main__":

    app.run(debug=True)
