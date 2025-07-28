import nltk
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from textstat import flesch_reading_ease
import spacy

nltk.download('reuters')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")

docs = []
labels = []
doc_ids = reuters.fileids()[:100]

for doc_id in doc_ids:
    text = reuters.raw(doc_id)
    cats = reuters.categories(doc_id)
    docs.append(text)
    labels.append(cats[0] if cats else 'unknown')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

doc_lengths = [len(doc.split()) for doc in docs]
all_words = []

for doc in docs:
    tokens = preprocess(doc)
    all_words.extend(tokens)

word_freq = Counter(all_words)

print(f"Total documents: {len(docs)}")
print(f"Average document length: {np.mean(doc_lengths):.2f} words")
print(f"Total unique words: {len(word_freq)}")
print(f"Top 10 most common words:")
for word, count in word_freq.most_common(10):
    print(f"  {word}: {count}")

def extract_entities(text):
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    return entities

def extract_patterns(text):
    patterns = {}
    
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\w+ \d{1,2}, \d{4}\b'
    dates = re.findall(date_pattern, text)
    patterns['dates'] = dates
    
    money_pattern = r'\$\d+(?:\.\d{2})?|\d+\.\d+ million|\d+\.\d+ billion'
    money = re.findall(money_pattern, text)
    patterns['money'] = money
    
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    patterns['phones'] = phones
    
    return patterns

all_entities = []
all_patterns = []

for doc in docs:
    entities = extract_entities(doc)
    patterns = extract_patterns(doc)
    all_entities.append(entities)
    all_patterns.append(patterns)

def tfidf_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text
    
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = preprocess(sentence)
        preprocessed_sentences.append(' '.join(tokens))
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    top_indices = sorted(top_indices)
    
    summary = ' '.join([sentences[i] for i in top_indices])
    return summary

def textrank_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text
    
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = preprocess(sentence)
        preprocessed_sentences.append(' '.join(tokens))
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    
    ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
    top_indices = [ranked_sentences[i][1] for i in range(num_sentences)]
    top_indices = sorted(top_indices)
    
    summary = ' '.join([sentences[i] for i in top_indices])
    return summary

tfidf_summaries = []
textrank_summaries = []

for doc in docs:
    tfidf_sum = tfidf_summary(doc)
    textrank_sum = textrank_summary(doc)
    tfidf_summaries.append(tfidf_sum)
    textrank_summaries.append(textrank_sum)

def evaluate(original_docs, summaries, method_name):
    compression_ratios = []
    readability_scores = []
    
    for orig, summ in zip(original_docs, summaries):
        orig_words = len(orig.split())
        summ_words = len(summ.split())
        compression_ratio = summ_words / orig_words if orig_words > 0 else 0
        compression_ratios.append(compression_ratio)
        
        try:
            readability = flesch_reading_ease(summ)
            readability_scores.append(readability)
        except:
            readability_scores.append(50)
    
    print(f"\n{method_name} Evaluation:")
    print(f"Average compression ratio: {np.mean(compression_ratios):.3f}")
    print(f"Average readability score: {np.mean(readability_scores):.2f}")
    print(f"Sample summary:\n{summaries[0][:200]}...")

evaluate(docs, tfidf_summaries, "TF-IDF")
evaluate(docs, textrank_summaries, "TextRank")

def search_docs(query):
    query_tokens = preprocess(query)
    relevant_docs = []
    
    for i, doc in enumerate(docs):
        doc_tokens = preprocess(doc)
        overlap = len(set(query_tokens) & set(doc_tokens))
        if overlap > 0:
            relevant_docs.append((i, overlap, doc))
    
    relevant_docs.sort(key=lambda x: x[1], reverse=True)
    return relevant_docs[:5]

def get_info(query):
    relevant_docs = search_docs(query)
    extracted_info = {}
    
    for doc_idx, _, _ in relevant_docs:
        entities = all_entities[doc_idx]
        patterns = all_patterns[doc_idx]
        summary = tfidf_summaries[doc_idx]
        
        extracted_info[doc_idx] = {
            'entities': entities,
            'patterns': patterns,
            'summary': summary
        }
    
    return extracted_info

def answer_query(query):
    print(f"\nProcessing query: '{query}'")
    
    relevant_docs = search_docs(query)
    print(f"Found {len(relevant_docs)} relevant documents")
    
    extracted_info = get_info(query)
    
    answer_parts = []
    all_entities_found = {}
    
    for doc_idx in extracted_info:
        summary = extracted_info[doc_idx]['summary']
        entities = extracted_info[doc_idx]['entities']
        
        answer_parts.append(summary)
        
        for entity_type, entity_list in entities.items():
            if entity_type not in all_entities_found:
                all_entities_found[entity_type] = []
            all_entities_found[entity_type].extend(entity_list)
    
    final_answer = "Based on the relevant documents:\n\n"
    final_answer += "\n\n".join(answer_parts[:3])
    
    if all_entities_found:
        final_answer += "\n\nKey entities found:\n"
        for entity_type, entities in all_entities_found.items():
            unique_entities = list(set(entities))[:5]
            final_answer += f"- {entity_type}: {', '.join(unique_entities)}\n"
    
    return final_answer

test_queries = [
    "financial markets",
    "company earnings", 
    "economic growth",
    "trade agreements"
]

print("AGENTIC SYSTEM DEMONSTRATION")
print("="*50)

for query in test_queries:
    answer = answer_query(query)
    print(answer)
    print("\n" + "-"*50)