import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, pipeline
import torch.nn.functional as F

# ===== ДАННЫЕ =====
data = pd.DataFrame({
    'post': [
        "Помогите с Python, не могу разобраться с циклом",
        "Ошибка NullPointerException в Java",
        "Как сделать адаптивный сайт на CSS и HTML",
        "Нужна помощь с Unity, сцена не работает",
        "Проблема с установкой библиотеки в Python",
        "Помогите с Spring Boot на Java"
    ],
    'category': ['Python', 'Java', 'Frontend', 'Unity', 'Python', 'Java'],
    'price': [100, 200, 150, 250, 180, 220]  # примерные цены
})

categories = data['category'].unique()
category_to_idx = {cat: i for i, cat in enumerate(categories)}
idx_to_category = {i: cat for cat, i in category_to_idx.items()}
labels = data['category'].map(category_to_idx).tolist()

# ===== ЭМБЕДДИНГИ =====
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
base_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = base_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


embeddings = torch.cat([get_embedding(post) for post in data['post']], dim=0)


# ===== ПЕРВАЯ СЕТЬ (классификация) =====
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # регуляризация
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ===== ВТОРАЯ СЕТЬ (регрессор цены) =====
class PriceRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# ===== ОБУЧЕНИЕ =====
input_dim = embeddings.shape[1]
num_classes = len(categories)

# Классификатор
net1 = SimpleClassifier(input_dim, num_classes)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(net1.parameters(), lr=0.001)

X = embeddings
y = torch.tensor(labels)

for epoch in range(50):  # умеренное количество эпох
    optimizer_cls.zero_grad()
    outputs = net1(X)
    loss = criterion_cls(outputs, y)
    loss.backward()
    optimizer_cls.step()
    if epoch % 10 == 0:
        print(f"[Classifier] Epoch {epoch + 1}/50, Loss: {loss.item():.4f}")

# Регрессор цены
prices = torch.tensor(data['price'].values, dtype=torch.float32)
net2 = PriceRegressor(input_dim)
criterion_reg = nn.MSELoss()
optimizer_reg = optim.Adam(net2.parameters(), lr=0.001)

for epoch in range(50):
    optimizer_reg.zero_grad()
    outputs = net2(X).squeeze()
    loss = criterion_reg(outputs, prices)
    loss.backward()
    optimizer_reg.step()
    if epoch % 10 == 0:
        print(f"[PriceRegressor] Epoch {epoch + 1}/50, Loss: {loss.item():.4f}")

# ===== ТОКСИЧНОСТЬ / СПАМ =====
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


# ===== СЛОЖНОСТЬ =====
TECH_TERMS = ["Python", "Java", "CSS", "HTML", "Unity", "Spring", "React", "Django", "Flask", "NumPy", "Pandas"]


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


# ===== ПОХОЖИЕ ПОСТЫ =====
def find_similar_posts(new_emb, embeddings, data, top_k=3):
    similarities = F.cosine_similarity(new_emb, embeddings)
    top_indices = similarities.topk(top_k).indices.tolist()
    return data.iloc[top_indices][["post", "category"]]


# ===== ОСНОВНОЙ ЦИКЛ =====
while True:
    new_post = input("\nВведите текст поста (или 'exit'): ")
    if new_post.lower() == 'exit':
        break

    new_emb = get_embedding(new_post)

    # Категория
    with torch.no_grad():
        out1 = net1(new_emb)
        probs = F.softmax(out1, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        predicted_category = idx_to_category[predicted_idx]

    # Цена
    with torch.no_grad():
        out2 = net2(new_emb)
        predicted_price = out2.item()

    # Сложность
    difficulty = estimate_difficulty_smart(new_post)

    # Спам/токсичность
    spam_flag = is_spam(new_post)

    # Похожие посты
    similar_posts = find_similar_posts(new_emb, embeddings, data)

    # Итог
    print("\n--- РЕЗУЛЬТАТ ---")
    print("Категория:", predicted_category)
    print("Вероятности категорий:", {idx_to_category[i]: f"{probs[0, i].item():.2f}" for i in range(len(categories))})
    print("Примерная цена:", round(predicted_price, 2), "₽")
    print("Сложность:", difficulty)
    print("Спам/Токсичный:", spam_flag)
    print("Похожие посты:")
    print(similar_posts)