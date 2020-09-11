import sys
import os
import shutil
import time
import traceback
import re
from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
# Uncomment to download "stopwords"
nltk.download("stopwords")
from nltk.corpus import stopwords




app = Flask(__name__)
#run_with_ngrok(app)


model_directory = 'model'
model_file_name = 'nbmodel.pickle'
tfidf_file_name = 'tfidfTransformer.pickle'

# These will be populated at training time
model_columns = None
clf = None


def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s



def model_prediction(review_text):
  processed_text = np.array([text_preprocessing(text) for text in review_text])
  test_vector = tfIDF.transform(processed_text)
  predictions = model.predict_proba(test_vector)
  predictions_label = []
  for p in predictions:
    if p[1]>0.75:
      rating = 5
      class_p = 'Positive'
    elif p[1]<0.75 and p[1]>0.50:
      rating = 4
      class_p = 'Positive'
    elif p[1]<0.50 and p[1]>0.25:
      rating = 2
      class_p = 'Negative'
    else:
      rating=1
      class_p = 'Negative'
    predictions_label.append({'Probablities':list(p),'Class':class_p,'Rating':rating})    
  return predictions_label



@app.route('/predict', methods=['POST']) # Create http://host:port/predict POST end point
def predict():
    if model:
        try:
            json_ = request.json #capture the json from POST
            print(json_)
            prediction = model_prediction(json_['reviews'])
            print(prediction)

            return {'Output':prediction}

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'

 

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    try:
        model = pickle.load(open(model_file_name,'rb'))
        print('model loaded')
        tfIDF = pickle.load(open(tfidf_file_name,'rb'))
        print('model columns loaded')

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        clf = None

    app.run(host='0.0.0.0', port=port, debug=False)
