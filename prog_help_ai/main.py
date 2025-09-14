import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import torch.nn as nn
import torch.optim as optim


# ДАТАСЕТ

data = pd.DataFrame({
    'post': [
        "Помогите с Python, не могу разобраться с циклом",
        "Ошибка NullPointerException в Java",
        "Как сделать адаптивный сайт на CSS и HTML",
        "Нужна помощь с Unity, сцена не работает",
        "Проблема с установкой библиотеки в Python",
        "Помогите с Spring Boot на Java"
    ],
    'category': ['Python', 'Java', 'Frontend', 'Unity', 'Python', 'Java']
})

categories = data['category'].unique()
category_to_idx = {cat: i for i, cat in enumerate(categories)}
idx_to_category = {i: cat for cat, i in category_to_idx.items()}
labels = data['category'].map(category_to_idx).tolist()


# ЭМБЕДДИНГИ

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

embeddings = torch.cat([get_embedding(post) for post in data['post']], dim=0)


# Нейросеть

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_dim = embeddings.shape[1]
num_classes = len(categories)
net = SimpleClassifier(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

X = embeddings
y = torch.tensor(labels)

# Обучение нейросети
for epoch in range(100):
    optimizer.zero_grad()
    outputs = net(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()


# Проверка токсичности / спама

toxic_classifier = pipeline("text-classification", model="unitary/toxic-bert")

def check_toxicity(post, threshold=0.5):
    result = toxic_classifier(post)[0]
    return result['label'] == 'TOXIC' and result['score'] >= threshold

def is_spam(post):
    if check_toxicity(post):
        return True
    if "http" in post.lower() or "www" in post.lower():
        return True
    if len(post.split()) < 3:
        return True
    return False


# Словарь технических терминов

TECH_TERMS = ["Python", "Java", "CSS", "HTML", "Unity", "Spring", "React", "Django", "Flask", "NumPy", "Pandas"]


# Улучшенная оценка сложности

def estimate_difficulty_smart(post):
    words = post.split()
    length = len(words)
    question_marks = post.count("?")
    unique_words = len(set(word.lower() for word in words))
    tech_count = sum(1 for term in TECH_TERMS if term.lower() in post.lower())
    
    score = 0
    if length > 15:
        score += 2
    elif length > 8:
        score += 1
    score += tech_count
    score += question_marks
    if unique_words > 10:
        score += 1
    
    if score >= 4:
        return "Сложный"
    elif score >= 2:
        return "Средний"
    else:
        return "Легкий"


# Поиск похожих постов

def find_similar_posts(new_emb, embeddings, data, top_k=3):
    similarities = torch.nn.functional.cosine_similarity(new_emb, embeddings)
    top_indices = similarities.topk(top_k).indices.tolist()
    return data.iloc[top_indices][["post", "category"]]

# Основной цикл

while True:
    new_post = input("\nВведите текст поста (или 'exit' для выхода): ")
    if new_post.lower() == 'exit':
        break

    # Категория
    new_emb = get_embedding(new_post)
    with torch.no_grad():
        output = net(new_emb)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_category = idx_to_category[predicted_idx]

    # Сложность
    difficulty = estimate_difficulty_smart(new_post)

    # Спам / токсичность
    spam_flag = is_spam(new_post)

    # Похожие посты
    similar_posts = find_similar_posts(new_emb, embeddings, data)

    # Результат
    print("\n--- РЕЗУЛЬТАТ ---")
    print("Категория:", predicted_category)
    print("Сложность:", difficulty)
    print("Спам/Токсичный:", spam_flag)
    print("Похожие посты:")
    print(similar_posts)