# A small flask based webviewer for images in a directory
# Usage: python img_viewer.py <path>
# Example: python img_viewer.py /home/user/Pictures

import os
import sys

from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    path = sys.argv[1]
    files = os.listdir(path)
    files = sorted(files)
    # chunk the files into 3 columns
    files = [files[i:i + 3] for i in range(0, len(files), 3)]
    with open('log.txt', 'r') as f:
        aselected_files = f.read().splitlines()
    selected_files = session.get('selected_files', aselected_files)
    return render_template('index.html', files=files, path=path,selected_files=selected_files)

@app.route('/discard', methods=['GET', 'POST'])
def discard():
    path = sys.argv[1]
    files = os.listdir(path)
    selected_files = request.form.getlist('discard')
    # Store the selected files in the session
    session['selected_files'] = selected_files
    with open('log.txt', 'a') as f:
        for file in selected_files:
            f.write(file + '\n')
    # chunk the files into 3 columns
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
