#Packages
from flask import Flask, render_template, request, redirect, Response
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

#Global app parameters
app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "C:/Users/longw/FlaskApp/static/images"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG", "JPG", "JPEG"]
app.config["results"] = "No Image Submitted"

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
PATH = "C:/Users/longw/FlaskApp/better_model.pt"

#Backend function
@app.route("/", methods=["GET", "POST"])
def upload_image():
    #check if something has been uploaded
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
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
            print("Image Saved")


            #prep the image
            img = Image.open(image).convert('RGB')
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
                
                newoutput = model(cropped_tensor)
                output = F.log_softmax(model(cropped_tensor), dim=1)
                _, pred = torch.max(output, dim=1)

            #analysis of the response of the model
            if pred[0] == 0:
                app.config["results"] = "This is CNV"
            else:
                if pred[0] == 1:
                    app.config["results"] = "This is DME"
                else:
                    if pred[0] == 2:
                        app.config["results"] = "This is Drusen"
                    else:
                        if pred[0] == 3:
                            app.config["results"] = "This is Normal"
        
    #return the answer to the html page
    return render_template("index.html", x = app.config["results"])

#debugger
if __name__ == "__main__":

    app.run(debug=True)
