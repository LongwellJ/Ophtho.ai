from flask import Flask, render_template, request, redirect

import os

from werkzeug.utils import secure_filename

from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
import cv2
import io

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


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
      
    
    
#defining my actual neural net
class NN(nn.Module):   
    def __init__(self):
        super(NN, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, 10)
        self.pool1 = nn.MaxPool2d(3,5)
        self.conv2 = nn.Conv2d(5, 5, 3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(6440, 500)
        self.fc2 = nn.Linear(500, 2)
        
        
        
    def forward(self, x):
        first = self.conv1(x)
        
        second = F.relu(first)

        third = self.pool1(second)

        fourth = self.conv2(third)

        fifth = F.relu(fourth)

        sixth = self.pool2(fifth)

        sixth = sixth.flatten(1)

        seventh = self.fc1(sixth)
        
        eighth = F.relu(seventh)
        
        nineth = self.fc2(eighth)
        
        
        return nineth


#Loading the model
PATH = "C:/Users/longw/FlaskApp/entire_model_newer.pt"


@app.route("/", methods=["GET", "POST"])
def upload_image():

    if request.files:

        if True:
            

                

            image = request.files["image"]
            in_memory_file = io.BytesIO()
            image.save(in_memory_file)
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(data, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (255,585))
            Tensor_d = torch.tensor(img, dtype=torch.float64)
            #new_tensor = torch.Tensor.np(Tensor_d)
            new_tensor = Tensor_d.reshape(1, 1,585,255)
            Tensor_d = torch.tensor(new_tensor)
            Tensor_d=Tensor_d.float()
            #print(Tensor_d)
            model = torch.load(PATH)
            model.eval()
            with torch.no_grad():
                
                newoutput = model(Tensor_d)
            print(newoutput)
            print(type(newoutput))
            
            if print(newoutput[0][0] > newoutput[0][1]) ==  torch.tensor(False):

                print("This is broken")

            else:
                print("This is not broken")

            

            print(newoutput[0][0]) 
            print(newoutput[0][1])
            

            
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
            
            
            
            return redirect("http://127.0.0.1:5000/results")

    return render_template("index.html")

@app.route("/results", methods=["GET", "POST"])
def display_results():


    return render_template("results.html")

if __name__ == "__main__":

    app.run(debug=True)
