#!/usr/bin/env python3
# from tryOn import TryOn as tryOn

from flask import Flask, render_template, Response,redirect,request
import os
from os import listdir
from os.path import isfile, join

app = Flask(__name__)

@app.route('/tryon/<file_path>',methods = ['POST', 'GET'])
def tryon(file_path):
	file_path = file_path.replace(',','/')
	os.system('python tryOn.py ' + file_path)
	return redirect('http://127.0.0.1:5000/',code=302, Response=None)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/checkout', methods=['GET'])
def checkout():
    return render_template('checkout.html')

@app.route('/compare', methods=['GET'])
def compare():
    path = './static/images/compare/'
    onlyfiles = [f for f in listdir(path)]
    print(onlyfiles)
    return render_template('compare.html', data = [path+s for s in onlyfiles])
    

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


if __name__ == '__main__':
    app.run()
    
