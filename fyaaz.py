# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 17:37:58 2021

@author: HP desktop
"""

from flask import Flask
app = Flask(__name__)
@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!, Mayy"
if __name__ == '__main__':
 #   
    app.run(debug=True,port=12345)
