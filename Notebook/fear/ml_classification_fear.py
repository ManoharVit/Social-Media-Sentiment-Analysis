
#Libraries

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

# Text processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import TweetTokenizer
from wordcloud import WordCloud
import re
# Download NLTK stopwords
nltk.download('stopwords')

# Machine learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Deep learning libraries
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Bidirectional, GRU, SpatialDropout1D, SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Display libraries
from IPython.display import display, Image
from PIL import Image as PILImage

# gen
import os
import numpy as np
import polars as pl
import altair as alt
import warnings as war
war.filterwarnings('ignore')

"""#DataProcessing"""

def load_and_rename(file_path, filename):
    full_path = f"{file_path}/{filename}"
    data = pl.read_csv(full_path, separator='\t')
    data.columns = ["ID", "Tweet", "Affect_Dimension", "Intensity_Class"]
    return data

# file paths'
file_path = '/Users/girimanoharv/Documents/Social-Media-Sentiment-Analysis/Dataset/English/EI-oc'
train_filename = 'training/EI-oc-En-fear-train.txt'
dev_filename = 'development/2018-EI-oc-En-fear-dev.txt'
test_filename = 'test-gold/2018-EI-oc-En-fear-test-gold.txt'

# Read datasets and rename columns
data_train = load_and_rename(file_path, train_filename)
data_dev = load_and_rename(file_path, dev_filename)
data_test = load_and_rename(file_path, test_filename)

print(data_train.head(2))
print(data_dev.head(2))
print(data_test.head(1))

# Concatenate training and development data
mer_data_train = pl.concat([data_train, data_dev])

# Mapping of categorical values to numeric codes
def generate_intensity_mapping(word):
    return {
        f"0: no {word} can be inferred": 0,
        f"1: low amount of {word} can be inferred": 1,
        f"2: moderate amount of {word} can be inferred": 2,
        f"3: high amount of {word} can be inferred": 3
    }

e_word = "fear"
intensity_mapping = generate_intensity_mapping("fear")


data_train = data_train.with_columns(
    pl.Series("Intensity_Class", [intensity_mapping[x] for x in data_train["Intensity_Class"]])
)

data_dev = data_dev.with_columns(
    pl.Series("Intensity_Class", [intensity_mapping[x] for x in data_dev["Intensity_Class"]])
)

data_test = data_test.with_columns(
    pl.Series("Intensity_Class", [intensity_mapping[x] for x in data_test["Intensity_Class"]])
)
mer_data_train = mer_data_train.with_columns(
    pl.Series("Intensity_Class", [intensity_mapping[x] for x in mer_data_train["Intensity_Class"]])
)

print(mer_data_train.head(3))
print(data_test.head(3))

#Intensity Class count occurrences with pie chart
intensity_counts = mer_data_train['Intensity_Class'].value_counts()

pie_chart = alt.Chart(intensity_counts).mark_arc(innerRadius=45).encode(
    theta=alt.Theta(field='count', type='quantitative', title='Count'),
    color=alt.Color(field='Intensity_Class', type='nominal', legend=alt.Legend(title="Intensity Class")),
    tooltip=['Intensity_Class', 'count']
).properties(
    title='Distribution of Intensity Classes'
).configure_title(
    fontSize=15,
    anchor='start'
)
pie_chart.display()

# Stop words
stop_words = set(stopwords.words('english'))
words_list = ['within', 'two', 'nine', 'zero', 'five', 'among', 'now', 'beside', 'seven', 'across', 'may', 'however', 'four', 'six', 'one', 'let', 'eight', 'three', 'ten', 'yet', 'also']
stop_words.update(words_list)
re_stop_words = re.compile(r"\b(" + "|".join(re.escape(word) for word in stop_words) + r")\b", re.I)

# Functions to Generate and display word clouds
def create_wordcloud(data, title):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stop_words,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1  # chosen at random
    ).generate(str(data))

    image_path = f"{title}.png"
    wordcloud.to_file(image_path)
    return image_path

def display_wordcloud(image_path, title):
    img = PILImage.open(image_path)
    display(img)

image_path = create_wordcloud(mer_data_train['Tweet'], f'Notebook/{e_word}/figures/Most_Common_{e_word}_Words')
display_wordcloud(image_path, 'Most Common Words from the Whole Chorpus')

image_path = create_wordcloud(mer_data_train.filter(pl.col('Intensity_Class') == 0)['Tweet'], f'Notebook/{e_word}/figures/No_{e_word}_Inferred')
display_wordcloud(image_path, 'No Anger Can Be Inferred')

image_path = create_wordcloud(mer_data_train.filter(pl.col('Intensity_Class') == 1)['Tweet'], f'Notebook/{e_word}/figures/Low_Amount_{e_word}_Inferred')
display_wordcloud(image_path, 'Low Amount of Anger Can Be Inferred')

image_path = create_wordcloud(mer_data_train.filter(pl.col('Intensity_Class') == 2)['Tweet'], f'Notebook/{e_word}/figures/Moderate_Amount_{e_word}_Inferred')
display_wordcloud(image_path, 'Moderate Amount of Anger Can Be Inferred')

image_path = create_wordcloud(mer_data_train.filter(pl.col('Intensity_Class') == 3)['Tweet'], f'Notebook/{e_word}/figures/High_Amount_{e_word}_Inferred')
display_wordcloud(image_path, 'High Amount of Anger Can Be Inferred')

def remove_stop_words(sentence):
    return re_stop_words.sub(" ", sentence)

mer_data_train = mer_data_train.with_columns(pl.col('Tweet').map_elements(remove_stop_words))
data_train = data_train.with_columns(pl.col('Tweet').map_elements(remove_stop_words))
data_test = data_test.with_columns(pl.col('Tweet').map_elements(remove_stop_words))
data_dev = data_dev.with_columns(pl.col('Tweet').map_elements(remove_stop_words))

data_test

stemmer = PorterStemmer()
# Stemming
stemmer = SnowballStemmer("english")
def stemming(sentence):
    return " ".join(stemmer.stem(word) for word in sentence.split())

data_train = data_train.with_columns(pl.col('Tweet').map_elements(stemming))
data_test = data_test.with_columns(pl.col('Tweet').map_elements(stemming))
data_dev = data_dev.with_columns(pl.col('Tweet').map_elements(stemming))
mer_data_train = mer_data_train.with_columns(pl.col('Tweet').map_elements(stemming))

data_test

# Combine train, dev and test tweet data for vectorization
tokenizer = TweetTokenizer()
tokenizer = nltk.tokenize.TreebankWordTokenizer()
full_text = list(mer_data_train['Tweet'].to_list()) + list(data_test['Tweet'].to_list())

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 1), tokenizer=tokenizer.tokenize)
vectorizer.fit(full_text)

train_vectorized = vectorizer.transform(data_train['Tweet'].to_list())
test_vectorized = vectorizer.transform(data_test['Tweet'].to_list())
dev_vectorized = vectorizer.transform(data_dev['Tweet'].to_list())
mer_vectorized = vectorizer.transform(mer_data_train['Tweet'].to_list())

vectorizer.transform(data_train['Tweet'])

"""#Machine learning Classification"""

# Extract the features and labels for training and test sets
x_train = mer_vectorized
y_train = mer_data_train['Intensity_Class'].to_numpy()
x_test = test_vectorized
y_test = data_test["Intensity_Class"].to_numpy()

# Train Logistic Regression model with One-vs-Rest strategy
lr = LogisticRegression()
ovr = OneVsRestClassifier(lr)
ovr.fit(x_train, y_train)

y_pred_ovr = ovr.predict(x_test)

accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
report_ovr = classification_report(y_test, y_pred_ovr)

print(f"Logistic Regression One-vs-Rest Classifier Accuracy: {accuracy_ovr}")
print(f"Logistic Regression One-vs-Rest Classifier Classification Report:\n{report_ovr}")

# RandomForest Classifier model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(x_train, y_train)
y_pred = rf_clf.predict(x_test)
print("Random ForestAccuracy:", accuracy_score(y_test, y_pred))
print(f"Random Forest Classification Report:\n{classification_report(y_test, y_pred)}")

# SVM Classifier model
svm = LinearSVC()
svm.fit(x_train, y_train)

y_pred_svm = svm.predict(x_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

print(f"SVM Classifier Accuracy: {accuracy_svm}")
print(f"SVM Classifier Classification Report:\n{report_svm}")

# Combine the models into a voting classifier
estimators = [('svm', svm), ('ovr', ovr)]
clf = VotingClassifier(estimators, voting='hard')
clf.fit(x_train, y_train)

y_pred_clf = clf.predict(x_test)

accuracy_clf = accuracy_score(y_test, y_pred_clf)
report_clf = classification_report(y_test, y_pred_clf)

print(f"Voting Classifier Accuracy: {accuracy_clf}")
print(f"Voting Classifier Classification Report:\n{report_clf}")

# Combine training data into a single string
train_text = ' '.join(mer_data_train['Tweet'].to_list())
wordcloud = WordCloud(width=800, height=600, background_color='white', min_font_size=10).generate(train_text)

# Save word cloud image
wc_img_path = f"Notebook/{e_word}/figures/wordcloud_train.png"
wordcloud.to_file(wc_img_path)
img = PILImage.open(wc_img_path)
display(img)

y_train = to_categorical(mer_data_train['Intensity_Class'].to_numpy())
y_val = to_categorical(data_test['Intensity_Class'].to_numpy())

# Parameters
max_features = 5000
max_words = 100
batch_size = 128
epochs = 10
num_classes = 4


# Prepare training and validation data
X_train = mer_data_train['Tweet'].to_list()
X_val = data_test['Tweet'].to_list()
Y_train = y_train
Y_val = y_val

# Print shapes to verify
print(f'X_train shape: {len(X_train)}')
print(f'Y_train shape: {Y_train.shape}')
print(f'X_val shape: {len(X_val)}')
print(f'Y_val shape: {Y_val.shape}')

# Tokenize the text data
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=max_words)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_words)

"""#CNN Model"""

# Define the model
CNN_model = Sequential()
CNN_model.add(Embedding(max_features, 100, input_length=max_words))
CNN_model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
CNN_model.add(MaxPooling1D(pool_size=2))
CNN_model.add(Flatten())
CNN_model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
CNN_model.add(Dropout(0.5))
CNN_model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
CNN_model.add(Dropout(0.5))
CNN_model.add(Dense(num_classes, activation='softmax'))
CNN_model.build(input_shape=(None, max_words))

# Compile the model with a smaller learning rate
CNN_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
CNN_model.summary()

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = CNN_model.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
loss, accuracy = CNN_model.evaluate(X_val_pad, y_val)
CNN_model.save(f"Notebook/{e_word}/CNN_model.h5")

print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

"""#RNN Model"""

# Define a RNN model
rnn_model = Sequential()
rnn_model.add(Embedding(max_features, 100, input_length=max_words))
rnn_model.add(SpatialDropout1D(0.2))
rnn_model.add(SimpleRNN(64, return_sequences=True))
rnn_model.add(Dropout(0.2))
rnn_model.add(SimpleRNN(64))
rnn_model.add(Dense(64, activation='relu'))
rnn_model.add(Dropout(0.5))
rnn_model.add(Dense(num_classes, activation='softmax'))
rnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
rnn_model.summary()
history_rnn = rnn_model.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
loss, accuracy = rnn_model.evaluate(X_val_pad, y_val)
rnn_model.save(f"Notebook/{e_word}/Rnn_model.h5")

"""#Bidirectional GRU"""

# Build the Bidirectional GRU model
model_BGRU = Sequential()
model_BGRU.add(Embedding(max_features, 100, input_length=max_words))
model_BGRU.add(SpatialDropout1D(0.25))
model_BGRU.add(Bidirectional(GRU(64, dropout=0.4, return_sequences=True)))
model_BGRU.add(Bidirectional(GRU(32, dropout=0.5, return_sequences=False)))
model_BGRU.add(Dense(num_classes, activation='sigmoid'))
# Explicitly build the model
model_BGRU.build(input_shape=(None, max_words))
model_BGRU.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
model_BGRU.summary()

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
historybgru = model_BGRU.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])

# Save Bi-directional GRU model
model_BGRU.save(f"Notebook/{e_word}/model_BGRU.h5")
loss, accuracy = model_BGRU.evaluate(X_val_pad, y_val)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

"""#LSTM model"""

# For LSTM; we use dev, train, and test
y_train = to_categorical(data_train['Intensity_Class'].to_numpy())
y_val = to_categorical(data_dev['Intensity_Class'].to_numpy())
y_test = to_categorical(data_test['Intensity_Class'].to_numpy())

X_train = data_train['Tweet'].to_list()
X_val = data_dev['Tweet'].to_list()
X_test = data_test['Tweet'].to_list()

# Tokenize text data
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(data_test['Tweet'])

# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_words)
X_val = pad_sequences(X_val, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)

class CustomLSTMModel(Model):
    def __init__(self, max_features, embedding_dim, num_classes):
        super(CustomLSTMModel, self).__init__()
        self.embedding = Embedding(max_features, embedding_dim, mask_zero=True)
        self.lstm1 = LSTM(64, dropout=0.4, return_sequences=True)
        self.lstm2 = LSTM(32, dropout=0.5, return_sequences=False)
        self.dense = Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        return self.dense(x)

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.layers.Input(shape=input_shape_nobatch)
        _ = self.call(inputs)

# Define and build the custom LSTM model
embedding_dim = 100
model3_LSTM = CustomLSTMModel(max_features, embedding_dim, num_classes)
model3_LSTM.build_graph((None, max_words))
model3_LSTM.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model3_LSTM.summary()
history = model3_LSTM.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
model3_LSTM.save(f"Notebook/{e_word}/model_LSTM.h5")
loss, accuracy = model3_LSTM.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Plot the model history
def plot_model_history(history):

# history.history dictionary to DataFrame
    history_df = pl.DataFrame({
        'epoch': range(1, len(history.history['accuracy']) + 1),
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    })

    history_df_long = history_df.melt(id_vars=['epoch'], value_vars=['accuracy', 'val_accuracy', 'loss', 'val_loss'], variable_name='type', value_name='value')

# Create charts
    accuracy_chart = alt.Chart(history_df_long.filter(pl.col('type').is_in(['accuracy', 'val_accuracy']))).mark_line().encode(
        x='epoch:Q',
        y='value:Q',
        color='type:N'
    ).interactive().properties(
        title='Model Accuracy'
    )

    loss_chart = alt.Chart(history_df_long.filter(pl.col('type').is_in(['loss', 'val_loss']))).mark_line().encode(
        x='epoch:Q',
        y='value:Q',
        color='type:N'
    ).interactive().properties(
        title='Model Loss'
    )

    return alt.hconcat(accuracy_chart, loss_chart)

plot_model_history(history)

"""#Model Testing"""

# Test the LSTM model
predict_class=model3_LSTM.predict(X_test)
tar_classes=np.argmax(predict_class,axis=1)

tar_classes.tolist()

