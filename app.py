from flask import Flask,render_template,request
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

model = pickle.load(open('sgd_gs.pkl','rb'))
tf = pickle.load(open('tf_sentimental.pkl','rb'))

stop_words = set(stopwords.words('english'))
stop_words.remove('very')
stop_words.remove('not')
stop_words.remove("isn't")
stop_words.remove("doesn't")
stop_words.remove("too")
stop_words.remove('most')



ps = PorterStemmer()


def preprocessing(a):
      a = re.sub('[^a-zA-Z]',' ',a)
      a = a.lower()
      a = a.split()
    
      message = [ps.stem(word) for word in a if not word in stop_words and len(word) >1]
      message = ' '.join(message)
      print(message)
      
      transformed = tf.transform([message])
      return transformed





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
