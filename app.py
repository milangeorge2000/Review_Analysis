from flask import Flask,render_template,request
import pickle

model = pickle.load(open('sgd_gs.pkl','rb'))

from preprocess import preprocessing


app = Flask(__name__)




@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def predict():
        message = request.form['textarea']
        prediction = model.predict(preprocessing(message))
        if prediction[0] == 0 :
            return render_template('index.html',info='This is a Negative Review')
        else:
            return render_template('index.html',info='This is a Positive Review')
   
    
           
    


if __name__ == '__main__':
    app.run(debug=True)
