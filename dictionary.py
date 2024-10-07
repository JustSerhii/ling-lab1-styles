import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import wikipediaapi
import csv
import string
from collections import defaultdict, Counter
import time
import os

# Завантаження необхідних ресурсів NLTK
nltk.download('brown')
nltk.download('punkt')
nltk.download('stopwords')

# Ініціалізація Wikipedia API з правильним User-Agent
wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent='EngStyleBot/1.0 (your.email@example.com)'  # Замініть на ваші дані
)

# Словники для кожного стилю
literary_words = []
scientific_words = []
news_words = []
technical_words = set()  # Використовуємо set для унікальності

# Розділи корпусу за стилями
literary_categories = ['fiction', 'romance']
scientific_categories = ['learned']
news_categories = ['news']

# Функція для збору технічних слів з Вікіпедії
def collect_technical_words_wikipedia(max_words=10000, delay=0.1):
    technical_categories_wiki = [
        'Computer_science',
        'Information_technology',
        'Engineering',
        'Software',
        'Hardware',
        'Artificial_intelligence',
        'Machine_learning',
        'Data_science',
        'Cybersecurity',
        'Blockchain',
        'Cloud_computing',
        'Internet_of_Things',
        'Quantum_computing',
        'DevOps',
        'Data_analysis',
        'Natural_language_processing',
        'Computer_vision',
        'Big_data',
        'Distributed_systems',
        'Virtualization',
        'Microservices',
        'Software_architecture',
        'Database_systems',
        'Computer_networks',
        'Operating_systems',
        'Programming_languages',
        'Software_development',
        'Web_development',
        'Mobile_computing',
        'Embedded_systems',
        'Computer_graphics',
        'Human_computer_interaction',
        'Computer_security',
        'Software_testing',
        'Data_mining',
        'Bioinformatics',
        'Robotics',
        'Computer_engineering',
        'Systems_engineering',
        'Information_systems',
        'Computer_science_theory',
        'Software_engineering'
    ]

    collected_words = set()
    for category in technical_categories_wiki:
        cat = wiki.page(f'Category:{category}')
        if not cat.exists():
            print(f"Категорія {category} не існує.")
            continue
        print(f"Збираємо сторінки з категорії: {category}")
        for subcat in cat.categorymembers.values():
            if subcat.ns == wikipediaapi.Namespace.MAIN and not subcat.title.startswith("Category:"):
                page = wiki.page(subcat.title)
                if page.exists():
                    text = page.text
                    words = word_tokenize(text)
                    words = [word.lower() for word in words if word.isalpha()]
                    words = [word for word in words if word not in stopwords.words('english')]
                    collected_words.update(words)
                    print(f"  Оброблено сторінку: {subcat.title} з {len(words)} словами.")
                    if len(collected_words) >= max_words:
                        print(f"Досягнуто {max_words} технічних слів.")
                        return set(list(collected_words)[:max_words])
                time.sleep(delay)  # Затримка для уникнення перевантаження API
    return collected_words

# Збираємо слова для кожного стилю з Brown Corpus
for category in literary_categories:
    literary_words.extend(brown.words(categories=category))

for category in scientific_categories:
    scientific_words.extend(brown.words(categories=category))

for category in news_categories:
    news_words.extend(brown.words(categories=category))

# Видаляємо дублікатні слова та перетворюємо на набір
literary_words = set([word.lower() for word in literary_words if word.isalpha()])
scientific_words = set([word.lower() for word in scientific_words if word.isalpha()])
news_words = set([word.lower() for word in news_words if word.isalpha()])

# Збираємо технічні слова з Вікіпедії
print("\nЗбираємо технічні слова з Вікіпедії...")
technical_words = collect_technical_words_wikipedia(max_words=10000, delay=0.1)
print(f"Зібрано {len(technical_words)} технічних слів.")

# Підготовка словника для CSV
dictionary = []

for word in literary_words:
    dictionary.append((word, "literary"))

for word in scientific_words:
    dictionary.append((word, "scientific"))

for word in news_words:
    dictionary.append((word, "news"))

for word in technical_words:
    dictionary.append((word, "technical"))

# Зберігаємо словник у CSV-файлі
with open('large_style_dictionary.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Word", "Style"])
    writer.writerows(dictionary)

print("Словник створено та збережено як 'large_style_dictionary.csv'.")

