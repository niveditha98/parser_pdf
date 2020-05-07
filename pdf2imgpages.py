# Import libraries 
from PIL import Image 
import pytesseract 
import sys 
from pdf2image import convert_from_path 
import os
from PyPDF2 import PdfFileReader, PdfFileWriter

def pdf_splitter(path):
        f = open(outfile, "a")
        pdf = open(path, "rb").read()
        startmark = b"\xff\xd8"
        startfix = 0
        endmark = b"\xff\xd9"
        endfix = 2
        i = 0

        njpg = 0
        while True:
                istream = pdf.find(b"stream", i)
                if istream < 0:
                        break
                istart = pdf.find(startmark, istream, istream+20)
                if istart < 0:
                        i = istream+20
                        continue
                iend = pdf.find(b"endstream", istart)
                if iend < 0:
                        raise Exception("Didn't find end of stream!")
                iend = pdf.find(endmark, iend-20)
                if iend < 0:
                        raise Exception("Didn't find end of JPG!")
                istart += startfix
                iend += endfix
                print("JPG ",njpg," from ",istart," to ",iend)
                jpg = pdf[istart:iend]
                jpgname='jpg'+str(njpg)+'.jpg'
                jpgfile = open("jpg%d.jpg" % njpg, "wb")
                jpgfile.write(jpg)
                 
                jpgfile.close()
                njpg += 1
                i = iend

# Path of the pdf 
PDF_file = sys.argv[1]

''' 
Part #1 : Converting PDF to images 
'''

# Store all the pages of the PDF in a variable 
pages = convert_from_path(PDF_file, 500) 

# Counter to store images of each page of PDF to image 
image_counter = 1

# Iterate through all the pages stored above 
for page in pages: 

	filename = PDF_file+"page_"+str(image_counter)+".jpg"
	
	# Save the image of the page in system 
	page.save(filename, 'JPEG') 

	# Increment the counter to update filename 
	image_counter = image_counter + 1


if(os.stat(filename).st_size == 0):
        pdf_splitter(PDF_file) 
