from gensim.models import Word2Vec
import pandas as pd

# Dataset
# https://www.kaggle.com/datasets/ankitkumar2635/sentiment-and-emotions-of-tweets

data = pd.read_csv('./dataset/sentiment-emotion-labelled_Dell_tweets.csv')
# Разделение данных на тренировочный и валидационный наборы (и, при необходимости, тестовый набор)
# Здесь можно выполнить дополнительную предварительную обработку данных, разделение и т. д.

# Получение количества строк в файле
num_rows = data.shape[0]

rows = data['Text']

# Создание пустого списка для сохранения слов
sentences = []

# Итерация по колонке и разбиение строк на слова
for row in rows:
    words = row.split()  # Разбиение строки на слова
    words_lower = [word.lower() for word in words]  # Приведение слов к нижнему регистру
    sentences.append(words_lower)  # Добавление списка слов в общий список

#print(word_list)

# Обучение модели Word2Vec
model = Word2Vec(sentences, min_count=1)  # Создаем и обучаем модель на предложениях

# Получение вектора для каждого слова
word_vectors = model.wv

# Примеры использования векторов
vector = word_vectors['billion']  # Получаем вектор для слова 'слово'
most_similar = word_vectors.most_similar('billion')  # Получаем наиболее похожие слова на 'слово'

# Пример использования векторов для предложения
#sentence = "Ваше предложение"  # Замените "Ваше предложение" на фактическое предложение, которое вы хотите преобразовать
#sentence_vector = sum([word_vectors[word] for word in sentence.split()]) / len(sentence.split())  # Преобразуем предложение в вектор путем усреднения векторов всех его слов

# Вывод результатов
print(vector)
print(most_similar)
#print(sentence_vector)