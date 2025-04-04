import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

articles = [
    {"id": 1, "title": "Climate Change Impacts", 
     "content": "Global warming is causing rising sea levels and extreme weather conditions."},
    {"id": 2, "title": "Renewable Energy Trends", 
     "content": "Solar and wind energy are becoming more prevalent as countries strive for sustainability."},
    {"id": 3, "title": "Extreme Weather Events", 
     "content": "Hurricanes and wildfires are increasingly linked to global warming."}
]

nlp = spacy.load("en_core_web_sm")
def preprocess(text):
    doc = nlp(text)
    keywords = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(keywords)

processed_articles = [preprocess(article["content"]) for article in articles]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_articles)

num_clusters = 2 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(X)

grouped_articles = {}
for label, article in zip(labels, articles):
    if label not in grouped_articles:
        grouped_articles[label] = []
    grouped_articles[label].append(article["title"])

for group, titles in grouped_articles.items():
    print(f"Group {group + 1}:")
    for title in titles:
        print(f"  - {title}")