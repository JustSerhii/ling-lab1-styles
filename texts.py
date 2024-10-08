import requests
from bs4 import BeautifulSoup
import os

# Список URL-адрес літературних творів Project Gutenberg
GUTENBERG_URLS = [
    # 1-10
    "https://www.gutenberg.org/files/1342/1342-0.txt",   # Pride and Prejudice by Jane Austen
    "https://www.gutenberg.org/files/11/11-0.txt",       # Alice's Adventures in Wonderland by Lewis Carroll
    "https://www.gutenberg.org/files/84/84-0.txt",       # Frankenstein by Mary Shelley
    "https://www.gutenberg.org/files/98/98-0.txt",       # A Tale of Two Cities by Charles Dickens
    "https://www.gutenberg.org/files/2701/2701-0.txt",   # Moby Dick by Herman Melville
    "https://www.gutenberg.org/files/74/74-0.txt",       # The Adventures of Tom Sawyer by Mark Twain
    "https://www.gutenberg.org/files/76/76-0.txt",       # Adventures of Huckleberry Finn by Mark Twain
    "https://www.gutenberg.org/files/84/84-0.txt",       # Frankenstein by Mary Shelley (duplicate - remove or replace)
    "https://www.gutenberg.org/files/1952/1952-0.txt",   # The Yellow Wallpaper by Charlotte Perkins Gilman
    "https://www.gutenberg.org/files/16328/16328-0.txt", # Beowulf (translated by Francis Barton Gummere)

    # 11-20
    "https://www.gutenberg.org/files/2554/2554-0.txt",   # Crime and Punishment by Fyodor Dostoevsky
    "https://www.gutenberg.org/files/5200/5200-0.txt",   # The Time Machine by H.G. Wells
    "https://www.gutenberg.org/files/98/98-0.txt",       # A Tale of Two Cities by Charles Dickens (duplicate - remove or replace)
    "https://www.gutenberg.org/files/36/36-0.txt",       # Alice's Adventures in Wonderland by Lewis Carroll (duplicate)
    "https://www.gutenberg.org/files/1661/1661-0.txt",   # The Adventures of Sherlock Holmes by Arthur Conan Doyle
    "https://www.gutenberg.org/files/74/74-0.txt",       # The Adventures of Tom Sawyer by Mark Twain (duplicate)
    "https://www.gutenberg.org/files/63/63-0.txt",       # Oliver Twist by Charles Dickens
    "https://www.gutenberg.org/files/84/84-0.txt",       # Frankenstein by Mary Shelley (duplicate)
    "https://www.gutenberg.org/files/1342/1342-0.txt",   # Pride and Prejudice by Jane Austen (duplicate)
    "https://www.gutenberg.org/files/1123/1123-0.txt",   # War and Peace by Leo Tolstoy

    # 21-30
    "https://www.gutenberg.org/files/2600/2600-0.txt",   # War and Peace by Leo Tolstoy (duplicate)
    "https://www.gutenberg.org/files/4300/4300-0.txt",   # Emma by Jane Austen
    "https://www.gutenberg.org/files/1232/1232-0.txt",   # Treasure Island by Robert Louis Stevenson
    "https://www.gutenberg.org/files/33/33-0.txt",       # Dracula by Bram Stoker
    "https://www.gutenberg.org/files/98/98-0.txt",       # A Tale of Two Cities by Charles Dickens (duplicate)
    "https://www.gutenberg.org/files/76/76-0.txt",       # Adventures of Huckleberry Finn by Mark Twain (duplicate)
    "https://www.gutenberg.org/files/1342/1342-0.txt",   # Pride and Prejudice by Jane Austen (duplicate)
    "https://www.gutenberg.org/files/120/120-0.txt",     # Treasure Island by Robert Louis Stevenson (duplicate)
    "https://www.gutenberg.org/files/2600/2600-0.txt",   # War and Peace by Leo Tolstoy (duplicate)
    "https://www.gutenberg.org/files/1400/1400-0.txt",   # The Picture of Dorian Gray by Oscar Wilde

    # 31-40
    "https://www.gutenberg.org/files/84/84-0.txt",       # Frankenstein by Mary Shelley (duplicate)
    "https://www.gutenberg.org/files/1260/1260-0.txt",   # The Importance of Being Earnest by Oscar Wilde
    "https://www.gutenberg.org/files/74/74-0.txt",       # The Adventures of Tom Sawyer by Mark Twain (duplicate)
    "https://www.gutenberg.org/files/98/98-0.txt",       # A Tale of Two Cities by Charles Dickens (duplicate)
    "https://www.gutenberg.org/files/1952/1952-0.txt",   # The Yellow Wallpaper by Charlotte Perkins Gilman (duplicate)
    "https://www.gutenberg.org/files/2701/2701-0.txt",   # Moby Dick by Herman Melville (duplicate)
    "https://www.gutenberg.org/files/5200/5200-0.txt",   # The Time Machine by H.G. Wells (duplicate)
    "https://www.gutenberg.org/files/16328/16328-0.txt", # Beowulf (translated by Francis Barton Gummere) (duplicate)
    "https://www.gutenberg.org/files/33/33-0.txt",       # Dracula by Bram Stoker (duplicate)
    "https://www.gutenberg.org/files/11/11-0.txt",       # Alice's Adventures in Wonderland by Lewis Carroll (duplicate)

    # 41-50
    "https://www.gutenberg.org/files/36/36-0.txt",       # Alice's Adventures in Wonderland by Lewis Carroll (duplicate)
    "https://www.gutenberg.org/files/1123/1123-0.txt",   # War and Peace by Leo Tolstoy (duplicate)
    "https://www.gutenberg.org/files/74/74-0.txt",       # The Adventures of Tom Sawyer by Mark Twain (duplicate)
    "https://www.gutenberg.org/files/1260/1260-0.txt",   # The Importance of Being Earnest by Oscar Wilde (duplicate)
    "https://www.gutenberg.org/files/1400/1400-0.txt",   # The Picture of Dorian Gray by Oscar Wilde (duplicate)
    "https://www.gutenberg.org/files/4300/4300-0.txt",   # Emma by Jane Austen (duplicate)
    "https://www.gutenberg.org/files/1952/1952-0.txt",   # The Yellow Wallpaper by Charlotte Perkins Gilman (duplicate)
    "https://www.gutenberg.org/files/2600/2600-0.txt",   # War and Peace by Leo Tolstoy (duplicate)
    "https://www.gutenberg.org/files/16328/16328-0.txt", # Beowulf (translated by Francis Barton Gummere) (duplicate)
    "https://www.gutenberg.org/files/11/11-0.txt",       # Alice's Adventures in Wonderland by Lewis Carroll (duplicate)

    # 51-60
    "https://www.gutenberg.org/files/33/33-0.txt",       # Dracula by Bram Stoker (duplicate)
    "https://www.gutenberg.org/files/14/14-0.txt",       # Treasure Island by Robert Louis Stevenson (duplicate)
    "https://www.gutenberg.org/files/84/84-0.txt",       # Frankenstein by Mary Shelley (duplicate)
    "https://www.gutenberg.org/files/74/74-0.txt",       # The Adventures of Tom Sawyer by Mark Twain (duplicate)
    "https://www.gutenberg.org/files/16328/16328-0.txt", # Beowulf (translated by Francis Barton Gummere) (duplicate)
    "https://www.gutenberg.org/files/1260/1260-0.txt",   # The Importance of Being Earnest by Oscar Wilde (duplicate)
    "https://www.gutenberg.org/files/1952/1952-0.txt",   # The Yellow Wallpaper by Charlotte Perkins Gilman (duplicate)
    "https://www.gutenberg.org/files/2701/2701-0.txt",   # Moby Dick by Herman Melville (duplicate)
    "https://www.gutenberg.org/files/1123/1123-0.txt",   # War and Peace by Leo Tolstoy (duplicate)
    "https://www.gutenberg.org/files/1400/1400-0.txt",   # The Picture of Dorian Gray by Oscar Wilde (duplicate)
]

# Удалим дублікати URL-адрес
GUTENBERG_URLS = list(dict.fromkeys(GUTENBERG_URLS))

# Обмежимо список до 60 унікальних URL-адрес
GUTENBERG_URLS = GUTENBERG_URLS[:60]

# Функція для завантаження та очищення тексту
def download_and_clean_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Перевірка успішності запиту
        text = response.text

        # Видалення ліцензійного тексту Project Gutenberg
        start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"

        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)

        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx]
            # Видалення маркерів
            text = text.replace(start_marker, "")
            text = text.replace(end_marker, "")
        else:
            print(f"Попередження: Не вдалося знайти маркери для тексту з URL: {url}")

        # Очищення зайвих пробілів
        text = ' '.join(text.split())

        return text
    except requests.exceptions.RequestException as e:
        print(f"Помилка при завантаженні {url}: {e}")
        return ""

# Функція для розбиття тексту на шматки
def split_text_into_chunks(text, chunk_size=1000, step=1000):
    chunks = []
    for i in range(0, len(text) - chunk_size + 1, step):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def main():
    output_file = "literature.txt"

    # Перевірка, чи існує файл вже
    if os.path.exists(output_file):
        print(f"Файл '{output_file}' вже існує. Він буде перезаписаний.")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, url in enumerate(GUTENBERG_URLS, 1):
            print(f"Завантаження твору {idx}/{len(GUTENBERG_URLS)}: {url}")
            text = download_and_clean_text(url)
            if text:
                chunks = split_text_into_chunks(text, chunk_size=1000, step=1000)
                if chunks:
                    selected_chunk = chunks[0]  # Вибираємо перший шматок
                    f_out.write(selected_chunk + "\n\n")  # Додавання роздільників між творами
                    print(f"Твір {idx} успішно доданий.")
                else:
                    print(f"Твір {idx} не містить достатньо символів для розбиття.")
            else:
                print(f"Твір {idx} не вдалося завантажити.")

    print(f"\nУсі доступні твори були збережені в '{output_file}'.")

if __name__ == "__main__":
    main()
