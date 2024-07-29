# Social-Media-Sentiment-Analysis
Evaluate six deep learning models on the Affects in Tweets Dataset, including CNN, Bidirectional GRU, LSTM, Logistic Regression, Support Vector Classifier, and a voting classifier. Achieve a peak accuracy and an overall good accuracy through thorough model assessment and testing.

## Project Structure

```plaintext

├── Dataset
│   ├── English
│   │   └── EI-oc
│   │       ├── development
│   │       │   ├── 2018-EI-oc-En-anger-dev.txt
│   │       │   ├── 2018-EI-oc-En-fear-dev.txt
│   │       │   ├── 2018-EI-oc-En-joy-dev.txt
│   │       │   └── 2018-EI-oc-En-sadness-dev.txt
│   │       ├── test-gold
│   │       │   ├── 2018-EI-oc-En-anger-test-gold.txt
│   │       │   ├── 2018-EI-oc-En-fear-test-gold.txt
│   │       │   ├── 2018-EI-oc-En-joy-test-gold.txt
│   │       │   └── 2018-EI-oc-En-sadness-test-gold.txt
│   │       └── training
│   │           ├── EI-oc-En-anger-train.txt
│   │           ├── EI-oc-En-fear-train.txt
│   │           ├── EI-oc-En-joy-train.txt
│   │           └── EI-oc-En-sadness-train.txt
│   ├── README.txt
│   └── SemEval2018-Task1-EN-data-description.pdf
├── LICENSE
├── Notebook
│   ├── Directory.py
│   ├── anger
│   │   ├── CNN_model.h5
│   │   ├── ML_Classification_anger.ipynb
│   │   ├── Rnn_model.h5
│   │   ├── figures
│   │   │   ├── High_Amount_anger_Inferred.png
│   │   │   ├── Low_Amount_anger_Inferred.png
│   │   │   ├── Moderate_Amount_anger_Inferred.png
│   │   │   ├── Most_Common_Words.png
│   │   │   ├── Most_Common_anger_Words.png
│   │   │   ├── No_anger_Inferred.png
│   │   │   ├── Word_cloud.png
│   │   │   └── wordcloud_train.png
│   │   ├── ml_classification_anger.py
│   │   ├── model_BGRU.h5
│   │   └── model_LSTM.h5
│   ├── fear
│   │   ├── CNN_model.h5
│   │   ├── ML_Classification_fear.ipynb
│   │   ├── Rnn_model.h5
│   │   ├── figures
│   │   │   ├── High_Amount_fear_Inferred.png
│   │   │   ├── Low_Amount_fear_Inferred.png
│   │   │   ├── Moderate_Amount_fear_Inferred.png
│   │   │   ├── Most_Common_fear_Words.png
│   │   │   ├── No_fear_Inferred.png
│   │   │   └── wordcloud_train.png
│   │   ├── ml_classification_fear.py
│   │   ├── model_BGRU.h5
│   │   └── model_LSTM.h5
│   ├── joy
│   │   ├── CNN_model.h5
│   │   ├── Rnn_model.h5
│   │   ├── figures
│   │   │   ├── High_Amount_joy_Inferred.png
│   │   │   ├── Low_Amount_joy_Inferred.png
│   │   │   ├── Moderate_Amount_joy_Inferred.png
│   │   │   ├── Most_Common_joy_Words.png
│   │   │   ├── No_joy_Inferred.png
│   │   │   └── wordcloud_train.png
│   │   ├── ml_classification_joy.py
│   │   ├── model_BGRU.h5
│   │   └── model_LSTM.h5
│   ├── reuse_code.py
│   ├── sadness
│   │   ├── CNN_model.h5
│   │   ├── Rnn_model.h5
│   │   ├── figures
│   │   │   ├── High_Amount_sadness_Inferred.png
│   │   │   ├── Low_Amount_sadness_Inferred.png
│   │   │   ├── Moderate_Amount_sadness_Inferred.png
│   │   │   ├── Most_Common_sadness_Words.png
│   │   │   ├── No_sadness_Inferred.png
│   │   │   └── wordcloud_train.png
│   │   ├── ml_classification_sadness.py
│   │   ├── model_BGRU.h5
│   │   └── model_LSTM.h5
│   └── test.py
├── README.md
└── Sentiment_Analysis_directory_structure.txt
