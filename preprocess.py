import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')





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
