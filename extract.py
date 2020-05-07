import csv
import os
from PIL import Image
import pytesseract

def cropimg(file):
  f = open(file)
  csv_f = csv.reader(f)

  #iterate through rows of your CSV
  for row in csv_f:

    #open image using PIL
    im = Image.open(row[4])
    #crop box format: xmin, ymin, xmax, ymax
    crop_box = (row[1], row[0], row[3], row[2])
    #convert values to int
    crop_box = map(int, crop_box)
    #crop using PIL
    img = im.crop((crop_box))
    #save the image to predetermined savepath
    img.save('cropped'+row[0])
  
