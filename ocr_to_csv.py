import csv
import io
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+")

def text_files_to_csv(files):
    rows = []
    for f in files:
        directory, filename = os.path.split(f)
        if os.stat(f).st_size != 0:
            with open(f) as of:
                txt = of.read()
        elif os.stat(f).st_size == 0:
            txt=''
        row, column = map(int, filename.split(".")[0].split("-"))
        '''print(row)
        print(column)'''
        if row == len(rows):
            rows.append([])
        rows[row].append(txt)

    csv_file = io.StringIO()
    writer = csv.writer(csv_file)
    writer.writerows(rows)
    text_file = open("csvval.txt", "w")
    text_file.write(csv_file.getvalue())
    text_file.close()
    return csv_file.getvalue()


def mainfn(files,doc):
    files.sort()
    s=text_files_to_csv(files)
    #print(s)
    ret=[]
    file=open('csvval.txt','r')
    for row_index, row in enumerate(file):
        #print(row)
        x=enumerate(row.split(","))
        for col_index, value in x:
            #print(value)
            b=value.strip()
            if b=='name' or b=='Name' or b=='NAME':
                ret.append(b+': '+x[col_index+1])
            elif 'name' in b or 'Name' in b or 'NAME' in b:
                ret.append(b)
    with open(r'final.csv', 'a') as f:
        for i,val in enumerate(ret):
            row=[]
            row.append(doc)
            row.append(val)
            writer = csv.writer(f)
            writer.writerow(row)
