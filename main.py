import nltk
import numpy as np
import csv
import string
import os
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

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
        # Використовуємо перше вимовляння слова в словнику
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


# Функція для підготовки текстів та їх міток
def prepare_texts_and_labels():
    texts = [
        # Художні тексти
        """The sun dipped below the horizon, painting the sky in hues of orange and pink. Birds chirped their evening songs as the gentle breeze rustled through the leaves. Emma stood on the porch, her thoughts drifting like the clouds overhead. She felt a sense of peace wash over her, a momentary escape from the chaos of the day.""",

        """The warm glow of the setting sun bathed the landscape in soft golden light, casting long shadows across the field. A gentle wind whispered through the tall grass, carrying with it the scent of wildflowers. Anna sat quietly on the hillside, watching as the day slowly gave way to twilight, her heart calm and her mind at ease, feeling completely in tune with the serenity of the world around her.""",

        """The old mansion stood silent at the end of the street, its windows dark and foreboding. Legends whispered of hidden passages and untold treasures, but most avoided it, fearing the unknown. Only the brave dared to explore its depths, driven by curiosity and the thrill of adventure.""",

        """Under the vast expanse of the night sky, twinkling stars illuminated the path ahead. Luna walked alone, her footsteps echoing in the stillness. She carried with her memories of days gone by, each step a reminder of the journey she had undertaken.""",

        """The garden was in full bloom, vibrant flowers swaying gently in the summer breeze. Bees buzzed from blossom to blossom, and butterflies danced in the sunlight. Mark tended to his plants with care, finding solace in the beauty that surrounded him.""",

        """The waves crashed against the rocky shore, the saltwater spray mixing with the cool ocean breeze. Sarah closed her eyes and breathed deeply, feeling the tension melt away. The rhythmic sound of the sea had always been her escape, a place where her mind could drift, free from the pressures of reality.""",

        """A lone candle flickered in the dark room, casting long shadows on the walls. The silence was thick, broken only by the occasional creak of the old wooden floor. Michael sat at the desk, quill in hand, pondering the words he would write, hoping they might carry the weight of the emotions he could not express aloud.""",

        """The rolling hills stretched out as far as the eye could see, covered in a blanket of wildflowers swaying in the breeze. Emma twirled through the meadow, her laughter echoing under the bright summer sun. For a moment, the world felt infinite, and the worries of life seemed distant and small.""",

        """A storm raged outside, rain lashing against the windows with an intensity that mirrored the turmoil in Jack's heart. He stared into the fire, the flames dancing in front of him, but his thoughts were elsewhere—lost in the memories of what could have been.""",

        """The forest was alive with the sounds of nature—the rustle of leaves, the chirping of birds, the distant gurgle of a stream. Maria wandered down the narrow path, her fingers brushing the bark of the ancient trees, feeling connected to something far greater than herself.""",

        """The crisp morning air filled the valley as the sun slowly climbed over the horizon. A light mist hovered above the grass, and in the distance, a deer grazed peacefully, unaware of the world’s complexities. Olivia stood still, letting the quiet serenity of nature wash over her.""",

        """Beneath the towering oak tree, Sophie read her favorite book, losing herself in its pages. The sound of rustling leaves accompanied her thoughts, and the warm afternoon sunlight filtered through the branches, casting playful shadows on the ground.""",

        """The rain drummed softly on the windowpane as David sat by the fire, a book in hand. Outside, the world was gray and cold, but within the walls of his cottage, there was warmth and comfort. His mind drifted between the pages, finding solace in the words.""",

        """The old market square bustled with life, vendors shouting their wares as people hurried by. Among the crowd, Eleanor moved with purpose, her eyes scanning the stalls for something that might catch her fancy, feeling the pulse of the city.""",

        """As the final notes of the piano faded into the stillness of the concert hall, the audience sat in silence, captivated by the music’s lingering echo. For one fleeting moment, time itself seemed to stop, and everyone was connected by the same melody.""",

        """The gentle rustle of leaves accompanied Lily as she strolled through the park. The late afternoon light bathed everything in a golden glow, and children’s laughter filled the air. She paused by the fountain, letting her thoughts drift away with the sound of the water.""",

        """The quiet streets were bathed in the soft glow of streetlights, casting long shadows on the cobblestones. Clara wandered aimlessly, her thoughts lost in the memories of a distant past, where everything seemed simpler and more carefree.""",

        """The scent of freshly baked bread filled the little bakery, its warmth seeping into the cold autumn air outside. Margaret stood behind the counter, watching as people came and went, each carrying a piece of comfort wrapped in parchment paper.""",

        """The ocean stretched out endlessly before him, its vast expanse shimmering under the midday sun. Tom stood at the edge of the cliff, feeling the wind whip through his hair, the salt air sharp on his skin, reminding him of the freedom that lay just beyond.""",

        """The fire crackled in the hearth as Emily sat in her grandmother’s old rocking chair, knitting a scarf. Outside, snowflakes fell silently, covering the world in a blanket of white. The warmth of the fire and the steady rhythm of her knitting needles brought her peace.""",

        # Наукові тексти
        """Recent advancements in renewable energy technologies have significantly improved the efficiency and cost-effectiveness of solar panels. Innovations in photovoltaic materials and energy storage systems are paving the way for widespread adoption, reducing dependency on fossil fuels and mitigating the impacts of climate change.""",

        """The study explores the behavioral patterns of urban wildlife in response to changing environmental conditions. Data collected over five years indicate a shift in migration routes and feeding habits, suggesting adaptability but also highlighting areas where human activity may be disrupting natural ecosystems.""",

        """Quantum entanglement remains one of the most intriguing phenomena in quantum mechanics. This research delves into the practical applications of entangled particles in secure communication systems, proposing a framework that could potentially revolutionize data transmission and encryption methods.""",

        """Genetic sequencing has transformed our understanding of hereditary diseases. By mapping the human genome, scientists can identify specific gene mutations responsible for various conditions, enabling the development of targeted therapies and personalized medicine approaches.""",

        """The integration of artificial intelligence in medical diagnostics has shown promising results. Machine learning algorithms can analyze complex datasets with high accuracy, assisting healthcare professionals in early disease detection and improving patient outcomes.""",

        """The global shift towards electric vehicles has led to significant advancements in battery technology. Researchers are exploring solid-state batteries as a promising alternative to traditional lithium-ion cells, offering higher energy densities and improved safety profiles.""",

        """A comprehensive study of climate change impacts on coral reefs has revealed alarming trends in ocean acidification. The data suggests that rising CO2 levels are contributing to the rapid degradation of coral ecosystems, threatening marine biodiversity on a global scale.""",

        """New developments in CRISPR technology are pushing the boundaries of genetic engineering. Scientists are now able to edit multiple genes simultaneously, paving the way for more complex modifications and opening up potential applications in agriculture, medicine, and biotechnology.""",

        """The exploration of exoplanets has entered a new era with the James Webb Space Telescope. By capturing infrared light, it allows astronomers to peer into distant planetary atmospheres, searching for signs of life in the universe beyond our solar system.""",

        """Recent studies on microplastics have shown their pervasive presence in marine and freshwater ecosystems. Researchers are investigating the long-term ecological effects and their potential bioaccumulation in the food chain, posing risks to both wildlife and human health.""",

        """The discovery of water ice on Mars has opened new possibilities for future human colonization. Scientists believe that this resource could provide not only drinking water but also oxygen and rocket fuel, drastically reducing the cost of interplanetary travel.""",

        """A recent breakthrough in gene editing technology has enabled scientists to correct genetic defects in living organisms. This advancement has the potential to cure hereditary diseases and could lead to significant developments in personalized medicine.""",

        """The study of black hole mergers has provided new insights into the nature of gravity and spacetime. Using data from gravitational wave detectors, researchers have been able to confirm predictions made by Einstein’s theory of general relativity.""",

        """Advances in quantum computing are pushing the boundaries of what is possible in fields such as cryptography, artificial intelligence, and material science. The ability to process complex calculations at unprecedented speeds could revolutionize industries.""",

        """A new model of climate change predicts more extreme weather events as global temperatures continue to rise. Researchers are working to refine these models to provide more accurate forecasts and to inform policy decisions regarding environmental protection.""",

        """The development of autonomous vehicles has seen rapid advancements in recent years. Machine learning algorithms allow these vehicles to navigate complex environments, avoiding obstacles and making real-time decisions without human intervention.""",

        """A comprehensive study on the effects of microgravity on human health has revealed significant changes in bone density, muscle mass, and immune function. These findings are crucial for preparing future long-term space missions.""",

        """Breakthroughs in materials science have led to the creation of new superconductor materials that operate at higher temperatures. This development could pave the way for more efficient energy transmission and advancements in medical imaging technologies.""",

        """The advent of synthetic biology has allowed scientists to design and create new biological systems from scratch. These engineered organisms have applications in medicine, agriculture, and environmental cleanup, presenting both opportunities and ethical challenges.""",

        """Recent studies on the brain’s plasticity have shown that the human brain can adapt and reorganize itself after injury. This finding has significant implications for neurorehabilitation and the treatment of conditions such as stroke and traumatic brain injury.""",

        # Технічні тексти
        """To deploy the new microservices architecture, ensure that each service is containerized using Docker. Utilize Kubernetes for orchestration to manage scaling and ensure high availability. Implement CI/CD pipelines to automate testing and deployment, minimizing downtime and streamlining the development process.""",

        """The API endpoints follow RESTful conventions, using standard HTTP methods for CRUD operations. Authentication is handled via OAuth 2.0, ensuring secure access. Documentation is provided using Swagger, allowing developers to understand and interact with the services effectively.""",

        """When configuring the network infrastructure, prioritize redundancy and failover mechanisms to maintain uptime. Use load balancers to distribute traffic evenly across servers, and implement monitoring tools to detect and respond to anomalies in real-time.""",

        """The software development lifecycle incorporates agile methodologies, facilitating iterative progress and continuous feedback. Utilize version control systems like Git to manage codebases, and leverage collaboration platforms such as GitHub for seamless teamwork and code reviews.""",

        """Database optimization involves indexing frequently queried fields to enhance retrieval speeds. Normalize tables to eliminate redundancy, and employ caching strategies using Redis to store transient data, reducing latency and improving overall application performance.""",

        """To enhance the security of the application, implement two-factor authentication (2FA) using time-based one-time passwords (TOTP). Integrate an open-source library for generating tokens, and ensure that user sessions are encrypted using TLS for all data transmissions.""",

        """The system architecture leverages a distributed database model, utilizing sharding to handle high volumes of data. Each shard is independently managed, allowing for horizontal scalability and improved performance under heavy loads.""",

        """When deploying the cloud infrastructure, ensure compliance with industry standards such as ISO/IEC 27001. Implement robust monitoring systems with real-time alerts for anomalies, and perform regular security audits to maintain the integrity of the platform.""",

        """The network topology follows a star configuration, with a central switch that manages all incoming and outgoing data flows. To minimize latency, use fiber-optic cables for backbone connections, and apply traffic prioritization techniques for critical services.""",

        """To improve fault tolerance, utilize a RAID 10 configuration for data storage, combining mirroring and striping. This setup provides both redundancy and enhanced read/write speeds, ensuring data availability even in the event of hardware failures.""",

        """To implement the new encryption algorithm, ensure that all communication between the client and server is secured using public-key cryptography. Utilize libraries such as OpenSSL to manage certificates and keys securely.""",

        """The cloud infrastructure should be designed to scale dynamically based on real-time demand. Using load balancers, auto-scaling groups, and container orchestration tools like Kubernetes can ensure that resources are efficiently allocated.""",

        """Ensure that the API adheres to RESTful principles, allowing for clear separation between client and server logic. All endpoints should use appropriate HTTP verbs (GET, POST, PUT, DELETE) and return status codes that accurately reflect the result of the operation.""",

        """The implementation of the new database schema involves normalizing all tables to third normal form (3NF). This will help reduce redundancy and ensure that data integrity is maintained throughout the application.""",

        """To optimize system performance, implement a caching layer using Redis or Memcached. This will reduce the load on the primary database and improve the response time for frequently accessed data.""",

        """The continuous integration pipeline should be configured to automatically run unit tests and static code analysis on each commit. This will ensure that any issues are caught early in the development process, reducing the likelihood of bugs reaching production.""",

        """When configuring the network for high availability, ensure that redundant failover systems are in place. Use multiple data centers and configure your DNS to direct traffic to the nearest location with active services.""",

        """For secure file transfers, use SFTP or FTPS protocols, which offer encryption and authentication. Avoid using unencrypted FTP, as it can expose sensitive data to interception during transmission.""",

        """Incorporate version control using Git for all development work. Establish branching strategies such as GitFlow to manage feature development, bug fixes, and releases in a structured manner.""",

        """Implement logging at various levels (info, warning, error) to monitor application health and identify potential issues. Use centralized logging services like ELK Stack to aggregate and analyze logs across different services.""",

        # Новинні тексти
        """The government today announced a new initiative aimed at reducing carbon emissions by 30% over the next decade. This initiative includes increased investment in renewable energy sources, stricter regulations on industrial emissions, and financial incentives for businesses that adopt greener practices.""",

        """In response to the recent surge in COVID-19 cases, health authorities have reintroduced mask mandates in public transportation and crowded indoor spaces. Experts emphasize the importance of vaccination and booster shots to curb the spread of the virus.""",

        """Tech giant XYZ unveiled its latest smartphone model today, boasting unprecedented battery life and an advanced camera system. Early reviews praise its sleek design and user-friendly interface, positioning it as a strong competitor in the highly saturated mobile market.""",

        """Economic indicators released this week show a steady recovery in the manufacturing sector, with a 5% increase in output compared to last quarter. Analysts attribute this growth to increased consumer demand and improved supply chain efficiencies.""",

        """The local community rallied together to support the annual food drive, donating over 10,000 pounds of non-perishable items. Organizers highlight the importance of such events in combating hunger and fostering a sense of solidarity among residents.""",

        """The city council has approved a new budget that allocates $50 million for public infrastructure improvements. The funds will be used to repair aging roads, expand public transportation, and enhance the city's green spaces.""",

        """In a historic ruling, the Supreme Court has struck down a key provision of the controversial law, citing constitutional concerns. The decision is expected to have wide-reaching implications for future cases involving civil liberties.""",

        """Local health officials are urging residents to get vaccinated ahead of the flu season, which is expected to be particularly severe this year. Free vaccination clinics will be set up across the city to encourage widespread immunization.""",

        """A major breakthrough in renewable energy was announced today as a team of engineers successfully developed a prototype for a solar-powered desalination plant. The new technology promises to provide fresh water to drought-stricken regions at a fraction of the current cost.""",

        """The stock market experienced a sharp drop this morning following reports of lower-than-expected earnings from several major tech companies. Analysts suggest that the market could face continued volatility in the coming weeks as economic uncertainty grows.""",

        """A new study has shown that air pollution levels in major cities around the world have increased significantly over the past decade, prompting governments to introduce stricter environmental regulations.""",

        """The mayor announced plans to build a new public transportation system that will reduce traffic congestion and cut down on carbon emissions. The project is expected to be completed within the next five years.""",

        """In a bid to address the growing housing crisis, local authorities have proposed new regulations that would require developers to include affordable housing units in all new residential projects.""",

        """The president signed a bill into law today that aims to boost economic growth by providing tax incentives for small businesses. The legislation is expected to create thousands of new jobs over the next few years.""",

        """Scientists have developed a new vaccine that has shown a 95% efficacy rate in clinical trials. The vaccine is expected to be distributed to the public in the coming months, following approval from regulatory bodies.""",

        """The local government has announced plans to revitalize the downtown area, investing in new infrastructure and public spaces. The initiative aims to attract more businesses and tourists to the city.""",

        """In response to the recent spike in crime rates, police have increased patrols in high-risk neighborhoods and are working closely with community leaders to address the root causes of violence.""",

        """The education ministry has unveiled a new curriculum aimed at improving students' digital literacy skills. The program will introduce coding and computer science courses in schools starting next year.""",

        """In the wake of recent natural disasters, emergency response teams are working around the clock to provide aid to affected communities. Rescue efforts are focused on delivering food, water, and medical supplies to the hardest-hit areas.""",

        """The stock market surged today, with major indexes reaching record highs. Analysts attribute the growth to strong corporate earnings reports and optimism about the global economic recovery.""",

    ]

    labels = ['literary'] * 20 + ['scientific'] * 20 + ['technical'] * 20 + ['news'] * 20
    return texts, labels

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


# Класифікація за допомогою SVM
def classify_with_svm(texts, labels, style_dict):
    # Створюємо матрицю ознак для кожного тексту
    X = create_feature_matrix(texts, style_dict)

    # Масштабуємо ознаки
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Розбиваємо на навчальну та тестову вибірку
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(X, labels, texts, test_size=0.8, random_state=42)

    # Виведення характеристик текстів
    print("Характеристики текстів:")
    for idx, text in enumerate(texts_test):
        features = extract_features(text, style_dict)
        print(f"\nТекст {idx + 1}:")
        print(f"  Кусок тексту: {text[:100]}...")
        print(f"  Середня довжина слів: {features[0]:.2f}")
        print(f"  Type-token ratio: {features[1]:.2f}")
        print(f"  Частка функціональних слів: {features[2]:.2f}")
        print(f"  Розподіл POS: NN={features[3]:.2f}, VB={features[4]:.2f}, JJ={features[5]:.2f}")
        print(f"  Flesch Reading Ease: {features[6]:.2f}")

    # Класифікація за допомогою SVM
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    # Прогнозування на тестовій вибірці
    y_pred = classifier.predict(X_test)

    # Виведення результатів класифікації
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Виведення результатів для кожного тексту
    print("\nРезультати для кожного тексту:")
    for idx, (text, true_label, pred_label) in enumerate(zip(texts_test, y_test, y_pred)):
        print(f"\nТекст {idx + 1}:")
        print(f"  Кусок тексту: {text[:100]}...")
        print(f"  Справжній стиль: {true_label}")
        print(f"  Прогнозований стиль: {pred_label}")

# Функція кластеризації
# def perform_clustering(feature_matrix, num_clusters=4):
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(feature_matrix)
#
#     # Використовуємо KMeans для кластеризації
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     labels = kmeans.fit_predict(scaled_features)
#
#     # Оцінка якості кластеризації
#     from sklearn.metrics import silhouette_score
#     if len(set(labels)) > 1:
#         silhouette_avg = silhouette_score(scaled_features, labels)
#         print(f"Середній коефіцієнт силуету: {silhouette_avg:.4f}")
#     else:
#         print("Неможливо обчислити Silhouette Score для одного кластера.")
#         silhouette_avg = 0
#
#     # Для візуалізації зменшимо розмірність до 2
#     pca = PCA(n_components=2, random_state=42)
#     components = pca.fit_transform(scaled_features)
#
#     return components, labels


# Функція для візуалізації кластерів
# def plot_clusters(components, labels):
#     plt.figure(figsize=(12, 8))
#     scatter = plt.scatter(components[:, 0], components[:, 1], c=labels, cmap='viridis', s=200, alpha=0.7)
#     plt.title('Кластеризація текстів на основі стилістичних ознак')
#     plt.xlabel('Компонента 1')
#     plt.ylabel('Компонента 2')
#     plt.grid(True)
#     plt.colorbar(scatter, ticks=range(len(set(labels))))
#     plt.show()


# Основна функція
def main():
    # Завантажуємо словник стилів
    style_dict = load_style_dictionary('large_style_dictionary.csv')
    if not style_dict:
        print("Словник стилів порожній або не завантажено. Деякі ознаки можуть бути відсутніми.")

    # Завантажуємо тексти та їх стилі
    texts, labels = prepare_texts_and_labels()

    # Виконуємо класифікацію за допомогою SVM
    classify_with_svm(texts, labels, style_dict)


if __name__ == "__main__":
    main()
