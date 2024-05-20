import flask
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load models at the top of the app to load into memory only one time
with open('models/loan_application_model_lr.pickle', 'rb') as f:
    clf_lr = pickle.load(f)

ss = StandardScaler()

genders_to_int = {'MALE': 1, 'FEMALE': 0}
married_to_int = {'YES': 1, 'NO': 0}
education_to_int = {'GRADUATED': 1, 'NOT GRADUATED': 0}
dependents_to_int = {'>=2': 0, '<2': 1}
self_employment_to_int = {'YES': 1, 'NO': 0}
property_area_to_int = {'RURAL': 0, 'URBAN': 1}

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/joinreport')
def report():
    return flask.render_template('jointreport.html')


@app.route('/loan')
def loan():
    return flask.render_template('loan.html')

@app.route("/Loan_Application", methods=['GET', 'POST'])
def Loan_Application():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')
    
    if flask.request.method == 'POST':
        # Get input from form
        genders_type = flask.request.form['genders_type']
        marital_status = flask.request.form['marital_status']
        dependents = flask.request.form['dependents']
        education_status = flask.request.form['education_status']
        self_employment = flask.request.form['self_employment']
        applicantIncome = float(flask.request.form['applicantIncome'])
        coapplicantIncome = float(flask.request.form['coapplicantIncome'])
        loan_amnt = float(flask.request.form['loan_amnt'])
        term_d = int(flask.request.form['term_d'])
        credit_history = int(flask.request.form['credit_history'])
        property_area = flask.request.form['property_area']

        # Create original output dict
        output_dict = {
            'Applicant Income': applicantIncome,
            'Co-Applicant Income': coapplicantIncome,
            'Loan Amount': loan_amnt,
            'Loan Amount Term': term_d,
            'Credit History': credit_history,
            'Gender': genders_type,
            'Marital Status': marital_status,
            'Education Level': education_status,
            'No of Dependents': dependents,
            'Self Employment': self_employment,
            'Property Area': property_area,
        }

        # Prepare input for prediction
        x = np.zeros(21)
        x[0] = applicantIncome
        x[1] = coapplicantIncome
        x[2] = loan_amnt
        x[3] = term_d
        x[4] = credit_history

        print('------this is array data to predict-------')
        print('X = ' + str(x))
        print('------------------------------------------')

        # Make prediction
        pred = clf_lr.predict([x])[0]
        result = 'Your Loan Application has been Approved!' if pred == 1 else 'Unfortunately, your Loan Application has been Denied'

        # Render different templates based on the prediction result
        if pred == 1:
            return flask.render_template('approved.html', result=result, original_input=output_dict)
        else:
            return flask.render_template('denied.html', result=result, original_input=output_dict)

if __name__ == '__main__':
    app.run(debug=True)
