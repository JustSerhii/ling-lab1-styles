import nltk
import numpy as np
import csv
import string
import os
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from imblearn.over_sampling import SMOTE  # Якщо потрібно балансування

matplotlib.use('TkAgg')

# Завантажуємо необхідні ресурси NLTK
nltk.download('punkt')
nltk.download('cmudict')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
cmu_dict = nltk.corpus.cmudict.dict()

# Функція для підрахунку кількості складів у слові
def syllable_count(word):
    word = word.lower()
    if word in cmu_dict:
        return len([y for y in cmu_dict[word][0] if y[-1].isdigit()])
    else:
        count = 0
        vowels = "aeiouy"
        if word and word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count

# Функція для розрахунку індексу Flesch Reading Ease
def flesch_reading_ease(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word for word in words if word not in punctuation]
    syllables = sum(syllable_count(word) for word in words)
    if len(sentences) == 0 or len(words) == 0:
        return 0
    ASL = len(words) / len(sentences)
    ASW = syllables / len(words)
    FRE = 206.835 - (1.015 * ASL) - (84.6 * ASW)
    return FRE

# Функція для завантаження словника стилів
def load_style_dictionary(filename='large_style_dictionary.csv'):
    style_dict = {}
    if not os.path.exists(filename):
        print(f"Файл {filename} не знайдено. Будь ласка, переконайтеся, що файл існує.")
        return style_dict
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            word = row['Word'].lower()
            style = row['Style'].lower()
            if word not in style_dict:
                style_dict[word] = style
    return style_dict

# Функція для розбиття тексту на менші частини без перекриття
# Функція для розбиття тексту на менші частини, ігноруючи всі зайві відступи '\n'
def load_and_split(filename, chunk_size=1000, step=1000):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    # Замінюємо всі нові рядки на пробіли
    text = text.replace('\n', ' ')
    # Видаляємо зайві пробіли
    text = ' '.join(text.split())
    chunks = [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size + 1, step)]
    # Переконайтеся, що вибірки унікальні
    unique_chunks = list(set(chunks))
    return unique_chunks


# Функція для завантаження всіх текстів і міток
def load_texts_from_files(filenames, styles, chunk_size=1000, step=1000):
    texts = []
    labels = []
    for style, filename in zip(styles, filenames):
        if not os.path.exists(filename):
            print(f"Файл {filename} не знайдено. Пропускаємо.")
            continue
        chunks = load_and_split(filename, chunk_size, step)
        texts.extend(chunks)
        labels.extend([style] * len(chunks))
    return texts, labels

# Функції для витягання ознак
def avg_word_length(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in punctuation]
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)

def type_token_ratio(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in punctuation]
    unique_words = set(words)
    if len(words) == 0:
        return 0
    return len(unique_words) / len(words)

def function_words_ratio(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in punctuation]
    function_word_count = sum(1 for word in words if word in stop_words)
    if len(words) == 0:
        return 0
    return function_word_count / len(words)

def pos_distribution(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in punctuation]
    tagged = nltk.pos_tag(words)
    counts = Counter(tag for word, tag in tagged)
    total = sum(counts.values())
    most_common_tags = ['NN', 'VB', 'JJ']  # Використовуємо лише 3 POS-теги
    distribution = [counts.get(tag, 0) / total if total > 0 else 0 for tag in most_common_tags]
    return distribution

# Функція для витягання ознак із словника стилів
def count_style_words(text, style_dict):
    words = word_tokenize(text.lower())
    style_counts = {'literary': 0, 'scientific': 0, 'news': 0, 'technical': 0}

    for word in words:
        if word in style_dict:
            style = style_dict[word]
            if style in style_counts:
                style_counts[style] += 1

    total_words = len(words)
    if total_words == 0:
        return [0, 0, 0, 0]

    return [
        style_counts['literary'] / total_words,
        style_counts['scientific'] / total_words,
        style_counts['news'] / total_words,
        style_counts['technical'] / total_words
    ]

# Витягання всіх ознак з тексту
def extract_features(text, style_dict):
    features = []
    features.append(avg_word_length(text))  # Середня довжина слів
    features.append(type_token_ratio(text))  # Type-token ratio
    features.append(function_words_ratio(text))  # Частка функціональних слів
    features.extend(pos_distribution(text))  # POS-теги
    style_word_ratios = count_style_words(text, style_dict)  # Співвідношення слів за стилями
    features.extend(style_word_ratios)
    features.append(flesch_reading_ease(text))  # Індекс читабельності Flesch
    return np.array(features)

# Масштабування ознак з вагами
def weight_features(features):
    weights = np.array([
        1.0,  # Avg Word Length
        1.0,  # Type-Token Ratio
        1.0,  # Function Words Ratio
        0.5,  # POS_NN
        0.5,  # POS_VB
        0.5,  # POS_JJ
        1.0,  # Style_Literary
        1.0,  # Style_Scientific
        1.0,  # Style_News
        1.0,  # Style_Technical
        1.0,  # Flesch Reading Ease
    ])
    if len(features) != len(weights):
        print(
            f"Кількість ваг ({len(weights)}) не відповідає кількості ознак ({len(features)}). Використовуються без ваг.")
        return features
    return features * weights

# Створення матриці ознак для текстів
def create_feature_matrix(texts, style_dict):
    feature_matrix = []
    for text in texts:
        features = extract_features(text, style_dict)
        weighted_features = weight_features(features)
        feature_matrix.append(weighted_features)
    return np.array(feature_matrix)

# Функція для візуалізації матриці змішування
def plot_confusion_matrix(cm, styles, fold):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=styles, yticklabels=styles, cmap='Blues')
    plt.title(f'Confusion Matrix for Fold {fold}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Функція для перевірки унікальності вибірок
def check_unique_samples(texts):
    unique_texts = set(texts)
    if len(unique_texts) != len(texts):
        print(f"Знайдено {len(texts) - len(unique_texts)} дублікатів.")
    else:
        print("Всі вибірки унікальні.")

# Функція для виводу розподілу класів
def print_class_distribution(y):
    counter = Counter(y)
    for cls, count in counter.items():
        print(f"Клас '{cls}': {count} вибірок")

# Ручна крос-валідація з матрицею змішування
def manual_cross_validation(X, y, styles, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1

    # Зберігаємо метрики для кожного фолду
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in skf.split(X, y):
        print(f"\nФолді {fold}:")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        # Тренуємо модель
        scaler = StandardScaler()
        classifier = SVC(kernel='linear')
        X_train_scaled = scaler.fit_transform(X_train)
        classifier.fit(X_train_scaled, y_train)

        # Тестуємо модель
        X_test_scaled = scaler.transform(X_test)
        y_pred = classifier.predict(X_test_scaled)

        # Обчислюємо метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # Зберігаємо метрики
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # Виводимо класифікаційний звіт
        print(f"Точність: {accuracy:.4f}")
        print(f"Прецизія (macro): {precision:.4f}")
        print(f"Відзив (macro): {recall:.4f}")
        print(f"F1-Score (macro): {f1:.4f}")
        print("Класифікаційний звіт:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Матриця змішування
        cm = confusion_matrix(y_test, y_pred, labels=styles)
        plot_confusion_matrix(cm, styles, fold)

        fold += 1

    # Обчислюємо середні значення та стандартні відхилення метрик
    print("\nСередні результати крос-валідації:")
    print(f"Точність: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Прецизія (macro): {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Відзив (macro): {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"F1-Score (macro): {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# Основна функція
def main():
    # Завантажуємо словник стилів
    style_dict = load_style_dictionary('large_style_dictionary.csv')
    if not style_dict:
        print("Словник стилів порожній або не завантажено. Деякі ознаки можуть бути відсутніми.")

    # Визначаємо файли та відповідні стилі
    styles = ['literary', 'scientific', 'technical', 'news']
    filenames = ['literary.txt', 'scientific.txt', 'technical.txt', 'news.txt']

    # Завантажуємо тексти та їх стилі з файлів
    texts, labels = load_texts_from_files(filenames, styles, chunk_size=500, step=500)
    print(f"Загальна кількість вибірок: {len(texts)}")

    if not texts:
        print("Немає текстів для аналізу. Перевірте наявність файлів та їхній вміст.")
        return

    # Перевіряємо унікальність вибірок
    check_unique_samples(texts)

    # Виводимо розподіл класів
    print("\nРозподіл класів:")
    print_class_distribution(labels)

    # Витягаємо ознаки для всіх текстів
    X = create_feature_matrix(texts, style_dict)
    y = labels

    print("Матриця ознак створена.")

    # (Опціонально) Балансування даних
    # smote = SMOTE(random_state=42)
    # X, y = smote.fit_resample(X, y)

    # Класифікація з використанням ручної крос-валідації
    manual_cross_validation(X, y, styles, n_splits=5)

if __name__ == "__main__":
    main()
