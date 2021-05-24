from pd import prob , prob_default , Z ,plot , box , cond
from Neur import NN
from flask import Flask , render_template, redirect, session, url_for, request
from flask_mysqldb import MySQL
import MySQLdb
import MySQLdb.cursors
import os
import re

app = Flask(__name__)
app.secret_key = "12345566"

#configuring the App
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "raghavvpandit"
app.config["MYSQL_DB"] = "login"

db = MySQL(app) #establishing connection to the database 

#route for login page
@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if 'uname' in request.form  and 'password' in request.form:
            uname = request.form['uname']
            password = request.form['password']
            cursor = db.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute("SELECT * FROM logininfo WHERE email=%s AND password=%s",(uname,password))
            info = cursor.fetchone()
            if info is not None:
              if info['email'] == uname and info['password'] == password:
                session['loginsuccess'] = True
                return redirect(url_for('profile'))
            else:
                return redirect(url_for('index'))
    return render_template('login.html')


#route for the register page
@app.route("/register",methods=['GET','POST'])
def new_user():
    if request.method  == 'POST':
        if 'one' in request.form and  'two' in request.form and 'three' in request.form:
            name = request.form['one']
            email = request.form['two']
            password = request.form['three']
            cur = db.connection.cursor(MySQLdb.cursors.DictCursor)      #connecting to the databse using cursor because the value can be sent and receive a dictionary
            cur.execute("INSERT INTO login.logininfo(name,password,email)VALUES(%s,%s,%s)",(name,password,email))
            db.connection.commit()
            return redirect(url_for('index'))
    return render_template('register.html')

app.config["FILE_UPLOAD"] = "./static"

#route for profile/home page   
@app.route("/profile",methods=['GET','POST'])
def profile():

    if request.method == "POST":
        check = request.form.getlist('check')
        req_output = ['pd' in check, 'fraud' in check]
        defval = 0
        if req_output[0]:
            if 'cred' in request.form and 'age' in request.form:
                cred_amt = request.form['cred']
                age = request.form['age']
                cred_amt = int(cred_amt)
                age = int(age)
            if request.files:
                ifile = request.files['inputfile']
                ifile.save(os.path.join(app.config["FILE_UPLOAD"],ifile.filename))

                defval = prob(ifile.filename,cred_amt,age)
                plot()
                box()
                cond()

        if req_output[1]:
            if request.files:

                ifile2 = request.files['inputfile2']
                ifile2.save(os.path.join(app.config["FILE_UPLOAD"],ifile2.filename))
                ifile3 = NN(ifile2.filename)
                #ifile3.to_csv('./static/result.csv')




        return render_template("output.html",defval = defval,req_output=req_output)

    if session['loginsuccess'] == True:
        return render_template('profile.html')

#Done so that server debugs and saves changes and restarts automatically
if  __name__ == '__main__':
    app.run(debug=True)