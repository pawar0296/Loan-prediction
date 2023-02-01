from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('LogModel.pickle','rb'))

app = Flask(__name__, template_folder="templates")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict_loan():
    Gender = int(request.form.get('Gender'))
    Married = int(request.form.get('Married'))
    Dependents = int(request.form.get('Dependents'))
    Education = int(request.form.get('Education'))
    Self_Employed = int(request.form.get('Self_Employed'))
    ApplicantIncome = int(request.form.get('ApplicantIncome'))
    CoapplicantIncome = int(request.form.get('CoapplicantIncome'))
    LoanAmount = int(request.form.get('LoanAmount'))
    Loan_Amount_Term = int(request.form.get('Loan_Amount_Term'))
    Credit_History = int(request.form.get('Credit_History'))
    Property_Area = int(request.form.get('Property_Area'))



    # prediction
    result = model.predict(np.array([Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]).reshape(1, -1))

    if result[0] == "N":
        result = 'NOT APPLICABLE'
    else:
        result = 'YES, APPLICABLE'

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8082)

print(__name__)