import os
import extract
import pandas as pd
import errno, stat, shutil
from extract_cells import main
from ocr_to_csv import mainfn
import csv
filepath="C:/Users/Admin/AppData/Local/Programs/Python/Python36/Lib/site-packages/example/"
doc=''
row=['File Name','Attribute (Name) in file']
with open("final.csv", "w") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(row)
    fp.close()
for file in os.listdir(filepath):
    if file.endswith(".pdf"):
        origpath=os.path.join(filepath, file)
        os.system('python pdf2imgpages.py '+origpath)
        os.system('python detection.py '+file)
        #give predicted results in csv format as input
        extract.cropimg(origpath+'.csv')
for file in os.listdir(filepath):
    if file.endswith('cropped.jpg'):
        myfile=file.split('.pdf')
        doc=myfile[0]+'.pdf'
        op=os.path.join(filepath,file)
        paths=main(op)
        print("\n".join(paths))
        argument=[]
        for f in os.listdir(filepath+"cells/"):
            opath=os.path.join(filepath+"cells/",f)
            os.system('python ocr_image.py '+opath)
            if f.endswith('.jpg'):
                p=os.path.join(filepath+'cells/ocr_data/',f)
                argument.append(p.replace('.jpg','.gt.txt') )
        #argument=argument.replace('.jpg','.gt.txt')
        mainfn(argument,doc)
        mydir=filepath+"cells/"
        filelist = [ f for f in os.listdir(mydir) if f.endswith('.jpg')]
        for f in filelist:
            os.remove(os.path.join(mydir, f))
        mydir1=filepath+'cells/ocr_data/'
        filelist1 = [ f for f in os.listdir(mydir1)]
        for f in filelist1:
            os.remove(os.path.join(mydir1, f))
os.rmdir(mydir1)
os.rmdir(mydir)
a=pd.read_csv('final.csv')
a.to_html('Final.htm')
        
