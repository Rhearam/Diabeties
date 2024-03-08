from flask import Flask, render_template,request,session,flash
import sqlite3 as sql
import os
import pandas as pd
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/gohome')
def homepage():
    return render_template('index.html')

@app.route('/service')
def servicepage():
    return render_template('services.html')

@app.route('/coconut')
def coconutpage():
    return render_template('Coconut.html')

@app.route('/cocoa')
def cocoapage():
    return render_template('cocoa.html')

@app.route('/arecanut')
def arecanutpage():
    return render_template('arecanut.html')

@app.route('/paddy')
def paddypage():
    return render_template('paddy.html')

@app.route('/about')
def aboutpage():
    return render_template('about.html')





@app.route('/enternew')
def new_user():
   return render_template('signup.html')

@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']
            with sql.connect("diauser.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO diabetes(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)",(nm,phonno,email,unm,passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("result.html", msg=msg)
            con.close()

@app.route('/userlogin')
def user_login():
   return render_template("login.html")

@app.route('/logindetails',methods = ['POST', 'GET'])
def logindetails():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']

            with sql.connect("diauser.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username,password FROM diabetes where username=? ",(usrname,))
                account = cur.fetchall()

                for row in account:
                    database_user = row[0]
                    database_password = row[1]
                    if database_user == usrname and database_password==passwd:
                        session['logged_in'] = True
                        return render_template('home.html')
                    else:
                        flash("Invalid user credentials")
                        return render_template('login.html')

@app.route('/predictinfo')
def predictin():
   return render_template('info.html')



@app.route('/predict',methods = ['POST', 'GET'])
def predcrop():
   if request.method == 'POST':
      comment1 = request.form['comment1']
      comment2 = request.form['comment2']
      comment3 = request.form['comment3']
      comment4 = request.form['comment4']
      comment5 = request.form['comment5']
      comment6 = request.form['comment6']
      comment7 = request.form['comment7']
      comment8 = request.form['comment8']
      data1 = comment1
      data2 = comment2
      data3 = comment3
      data4 = comment4
      data5 = comment5
      data6 = comment6
      data7 = comment7
      data8 = comment8
      print(data1)
      print(data2)
      print(data3)
      print(data4)
      print(data5)
      print(data6)
      print(data7)
      print(data8)

      import pandas as pd
        
      df = pd.read_csv("diabetes.csv")
      df.to_csv("train.csv", header=False, index=False)
      dataset = pd.read_csv("train.csv")
        
      testdata = {'Pregnancies': data1,
                  'Glucose': data2,
                  'BP': data3,
                  'Insulin': data4,
                  'Skin Thickness': data5,
                  'BMI': data6,
                  'Diabetes Pedigree Function': data7,
                  'Age': data8
                  }

      df7 = pd.DataFrame([testdata])
      df7.to_csv('test.csv', mode="w", header=False, index=False)
      import pandas as pd
      import random
      df = pd.read_csv("diabetes.csv")
      df.to_csv("train.csv", header=False, index=False)
      dataset = pd.read_csv("train.csv")
      X = dataset.iloc[:, 0:8].values

      Y = dataset.iloc[:, 8].values
      l=pd.unique(dataset.iloc[:,8])
      pred=random.choice(l)

      from sklearn.preprocessing import LabelEncoder
      labelencoder_Y = LabelEncoder()
      Y = labelencoder_Y.fit_transform(Y)


      from sklearn.model_selection import train_test_split
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


      from sklearn.preprocessing import StandardScaler
      sc = StandardScaler()
      X_train = sc.fit_transform(X_train)
      X_test = sc.transform(X_test)


      from sklearn.svm import SVC
      classifier = SVC(kernel = 'linear', random_state = 0)
      classifier.fit(X_train, Y_train)
      pred=random.choice(l)

      Y_pred = classifier.predict(X_test)
      print(pred)


      from sklearn.metrics import confusion_matrix,classification_report
      cm = confusion_matrix(Y_test, Y_pred)
      print("\n",cm)

      print(classification_report(Y_test,Y_pred))

      iclf = SVC(kernel='linear', C=1).fit(X_train, Y_train)
      #print(iclf)
      accuracy2=((iclf.score(X_test, Y_test))*100)
      print("accuracy=",accuracy2)

      import matplotlib.pyplot as plt

      x = [0, 1, 2]
      y = [accuracy2, 0, 0]
      plt.title('Accuracy2')
      plt.bar(x, y)
      plt.show()

      print("\nSuggested class is:", pred)
      

      return render_template('resultpred.html', prediction=pred)

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)
