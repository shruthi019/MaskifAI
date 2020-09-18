from flask import render_template, Response, redirect, flash, request, url_for
from app import app, db, mail
from app.forms import LoginForm, RegistrationForm
from app.camera import VideoCamera
from app.models import User
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.urls import url_parse
import time
from datetime import datetime
from flask_mail import Message
import sqlite3
import smtplib, ssl
from email.message import EmailMessage


@app.route('/') 
@app.route('/index')
def index():
    return render_template('index.html')
   
@login_required
def gen(camera):
    curr_time = time.time()
    while time.time() < curr_time + 5:
        frame, wearmask = camera.get_frame()
        global curr_datetime
        curr_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        global maskTF 
        maskTF = wearmask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen(VideoCamera()), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/success')
@login_required
def success():
    current_user.wearing_mask = maskTF 
    current_user.date_time = curr_datetime
    db.session.commit()
    return render_template('success.html')

@app.route('/check')
@login_required
def check():
    return render_template('maskdetection.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/covid_info')
def covid_info():
    return render_template('precaution.html')
 
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('You are registered!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/sendmail')
@login_required
def sendmail():
    if current_user.id == 1:
        conn = sqlite3.connect('app/app.db')
        cursor = conn.cursor()
        current_date = datetime.now().strftime('%d/%m/%Y %H:%M:%S')[:10] + '%'
        query = 'select id, username from user where wearing_mask = 0 and date_time like ?'
        cursor.execute(query, (current_date,))
        rows = cursor.fetchall()
        defaulters = []
        SPACE = '                         ' 
        for i in rows:
            defaulters.append(SPACE.join(str(j) for j in i))
        msg = Message('Alert from MaskifAI', sender = 'maskifai@gmail.com', recipients = ['srishtinegi925@gmail.com', 'raoshruthi2001@gmail.com', 'sachy01pm@gmail.com'])
        msg.body = 'Greetings from MaskifAI! \nHere are the employees who did not wear a mask today: \nEmployee_ID      Username\n' + '\n'.join(i for i in defaulters)
        with app.app_context():
            mail.send(msg)
        conn.commit()
        conn.close()
    else:
        flash('You are not authorised to send mail.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)