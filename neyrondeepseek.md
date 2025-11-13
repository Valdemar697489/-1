# 📁 Проект RAG Task

Простая реализация Retrieval-Augmented Generation (RAG) на Python.

---

## 🧩 Содержание
- `md_to_vectors.py`
- `vectorize_example_2.py`
- `try_example.py`
- `requirements.txt`
- `tz.md`
- `README.md`

---

## 📄 requirements.txt

```text
openai
faiss-cpu
numpy
tqdm

```

## 📄 md_to_vectors.py

```python
# md_to_vectors.py
# Скрипт для разбиения текста (например, ТЗ) на блоки и векторизации

import openai
import numpy as np
import faiss
from tqdm import tqdm

openai.api_key = "YOUR_API_KEY"  # замените на свой ключ

def read_text(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def embed_texts(texts):
    vectors = []
    for t in tqdm(texts, desc="Векторизация"):
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=t
        )
        vec = np.array(response.data[0].embedding, dtype="float32")
        vectors.append(vec)
    return np.array(vectors)

if __name__ == "__main__":
    text = read_text("tz.md")  # исходный файл с ТЗ
    chunks = split_text(text)
    vectors = embed_texts(chunks)

    # сохраняем векторное пространство
    faiss_index = faiss.IndexFlatL2(vectors.shape[1])
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, "vectors.index")

    print(f"✅ Векторизация завершена! Сохранено {len(vectors)} блоков.")

```

## 📄 vectorize_example_2.py

```python
# vectorize_example_2.py
# Поиск похожих векторов по запросу

import openai
import numpy as np
import faiss

openai.api_key = "YOUR_API_KEY"

def get_vector(query):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding, dtype="float32")

def search_similar(query, top_k=3):
    index = faiss.read_index("vectors.index")
    vec = get_vector(query)
    vec = np.expand_dims(vec, axis=0)

    distances, indices = index.search(vec, top_k)
    print("🔍 Похожие блоки:")
    print("Индексы:", indices[0])
    print("Дистанции:", distances[0])

if __name__ == "__main__":
    q = input("Введите запрос: ")
    search_similar(q)

```

## 📄 try_example.py

```python
# try_example.py
# Отправка запроса в нейросеть с найденными контекстами

import openai
import numpy as np
import faiss

openai.api_key = "YOUR_API_KEY"

def get_vector(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

def load_chunks():
    with open("tz.md", "r", encoding="utf-8") as f:
        text = f.read()
    return text.split("\n")

def find_similar_chunks(query, top_k=3):
    index = faiss.read_index("vectors.index")
    query_vec = get_vector(query)
    D, I = index.search(np.expand_dims(query_vec, 0), top_k)
    chunks = load_chunks()
    return [chunks[i] for i in I[0] if i < len(chunks)]

def ask_gpt(context, query):
    full_prompt = f"Контекст:\n{context}\n\nВопрос:\n{query}"
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    q = input("Введите запрос: ")
    similar_chunks = find_similar_chunks(q)
    context = "\n".join(similar_chunks)
    answer = ask_gpt(context, q)
    print("\nОтвет нейросети:\n")
    print(answer)

```

## 📄 README.md

```text
# KT11 – Генерация с RAG по существующему ТЗ

Простая реализация Retrieval-Augmented Generation (RAG).

## Файлы

- **md_to_vectors.py** — разбивает `tz.md` на блоки и векторизует.
- **vectorize_example_2.py** — ищет похожие блоки по запросу.
- **try_example.py** — отправляет запрос в модель с найденным контекстом.

## Использование

1. Создайте файл `tz.md` с текстом вашего ТЗ.
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Сначала запустите векторизацию:
   ```bash
   python md_to_vectors.py
   ```
4. Затем попробуйте поиск похожих фрагментов:
   ```bash
   python vectorize_example_2.py
   ```
5. И наконец — генерацию ответа:
   ```bash
   python try_example.py
   ```

> 💡 Всё максимально просто, сделано в учебных целях.

```

## 📄 tz.md

```text
# Техническое задание (пример)

## Цель проекта
Создать систему, которая помогает пользователям получать ответы на вопросы по документации.

## Основные задачи
- Разделить текст на блоки.
- Создать векторное представление блоков.
- При запросе находить наиболее похожие блоки.
- Использовать найденные блоки как контекст для генерации ответа.

## Требования
- Язык программирования: Python
- Используемые библиотеки: openai, faiss, numpy
- Результаты сохраняются в виде файла с векторами.

## Пример использования
Пользователь вводит вопрос, например: "Как происходит векторизация?"
Система ищет похожие блоки и формирует ответ с помощью модели GPT.

```

