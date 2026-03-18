from flask import (Flask, flash, redirect, render_template, request, session, url_for)
app = Flask(__name__)

@app.route('/Home')
def hello_world():
    return 'Hello, World!'

@app.route('/Profile')
def profile():
    return render_template('A1.html')
if __name__ == '__main__':
    app.run(debug=True) 

# This is a sample Python script.
#command to run the server "flask --app app.py run -- debug" in the terminal