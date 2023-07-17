from gensim.models import Word2Vec
import pandas as pd

# Dataset
# https://www.kaggle.com/datasets/ankitkumar2635/sentiment-and-emotions-of-tweets

data = pd.read_csv('./dataset/sentiment-emotion-labelled_Dell_tweets.csv')
# Разделение данных на тренировочный и валидационный наборы (и, при необходимости, тестовый набор)
# Здесь можно выполнить дополнительную предварительную обработку данных, разделение и т. д.

# Получение количества строк в файле
num_rows = data.shape[0]

text_rows = data['Text']

# Создание пустого списка для сохранения слов
sentences = []

# Итерация по колонке и разбиение строк на слова
for row in text_rows:
    words = row.split()  # Разбиение строки на слова
    words_lower = [word.lower() for word in words]  # Приведение слов к нижнему регистру
    sentences.append(words_lower)  # Добавление списка слов в общий список

#print(sentences)

# Обучение модели Word2Vec
model = Word2Vec(sentences, vector_size=200, min_count=1)  # Создаем и обучаем модель на предложениях

# Получение вектора для каждого слова
word_vectors = model.wv

# Примеры использования векторов
vector = word_vectors['billion']  # Получаем вектор для слова 'слово'
most_similar = word_vectors.most_similar('billion')  # Получаем наиболее похожие слова на 'слово'

# Пример использования векторов для предложения
#sentence = "Ваше предложение"  # Замените "Ваше предложение" на фактическое предложение, которое вы хотите преобразовать
#sentence_vector = sum([word_vectors[word] for word in sentence.split()]) / len(sentence.split())  # Преобразуем предложение в вектор путем усреднения векторов всех его слов

# Вывод результатов
#print(vector)
#print(most_similar)
#print(sentence_vector)

sentence_vectors = []

index = 0
stop = 5000
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
sentiments_rows = data['sentiment']

# Replace the values in the Sentiments column
sentiments_rows.replace({'positive': 2, 'negative': 1, 'neutral': 0}, inplace=True)
#print(sentiments_rows)

# Combine the sentiments and sentence_vectors lists
combined_list = list(zip(sentence_vectors, sentiments_rows))

# Write the combined list to a new CSV file
df = pd.DataFrame(combined_list, columns=['vector', 'sentiment'])
df.to_csv('./dataset/output.csv', index=True)