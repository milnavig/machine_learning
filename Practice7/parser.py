from gensim.models import Word2Vec
import pandas as pd

# Dataset
# https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_train.csv

data = pd.read_csv('./dataset/Corona_NLP_train.csv', encoding='latin1')
# Разделение данных на тренировочный и валидационный наборы (и, при необходимости, тестовый набор)
# Здесь можно выполнить дополнительную предварительную обработку данных, разделение и т. д.

# Получение количества строк в файле
num_rows = data.shape[0]

text_rows = data['OriginalTweet']

# Создание пустого списка для сохранения слов
sentences = []

# Итерация по колонке и разбиение строк на слова
for row in text_rows:
    words = row.split()  # Разбиение строки на слова
    words_lower = [word.lower() for word in words]  # Приведение слов к нижнему регистру
    sentences.append(words_lower)  # Добавление списка слов в общий список

# Обучение модели Word2Vec
model = Word2Vec(sentences, vector_size=200, min_count=1)  # Создаем и обучаем модель на предложениях

# Получение вектора для каждого слова
word_vectors = model.wv

sentence_vectors = []

index = 0
stop = 2000
for sentence in sentences:
    index = index + 1
    if index == stop:
        break
    sentence_vector = []
    for word in sentence:
        if word_vectors[word] is not None:
            vector = word_vectors[word].tolist()
            sentence_vector.append(vector)
    sentence_vectors.append(sentence_vector)

#print(sentence_vectors)

# Get the Sentiments column
sentiments_rows = data['Sentiment']

# Replace the values in the Sentiments column
sentiments_rows.replace({'Extremely Positive': 4, 'Positive': 3, 'Neutral': 2, 'Negative': 1, 'Extremely Negative': 0}, inplace=True)
#print(sentiments_rows)

# Combine the sentiments and sentence_vectors lists
combined_list = list(zip(sentence_vectors, sentiments_rows))

# Write the combined list to a new CSV file
df = pd.DataFrame(combined_list, columns=['vector', 'sentiment'])
df.to_csv('./dataset/output.csv', index=True)