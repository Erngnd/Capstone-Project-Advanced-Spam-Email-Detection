import email
from email import policy
from email.parser import BytesParser
import re
import requests
import numpy as np
import hashlib
import os
from html2text import html2text
from dotenv import load_dotenv
import pickle
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import os.path
#import string
#import random

load_dotenv(dotenv_path="C:/Users/Administrator/Downloads/sn.env")

# Spam keywords - expanded list
SPAM_KEYWORDS = [
    "win", "free", "offer", "prize", "click here", "congratulations", "limited time",
    "act now", "discount", "cheap", "cash", "loan", "credit", "investment", "guarantee",
    "urgent", "important", "alert", "notification", "account", "verify", "update",
    "validate", "confirm", "information", "password", "login", "security", "bank",
    "restricted", "warning", "attention", "payment", "transfer", "fund", "million",
    "bitcoin", "crypto"
]

# Create a basic vectorizer and model for spam detection
# This will be used to train a simple model if no pre-trained model exists
def create_basic_model():
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    # classifier = MultinomialNB(alpha=0.1)
    classifier = LogisticRegression(max_iter=1000) 
    # classifier = KNeighborsClassifier(n_neighbors=5)
    return vectorizer, classifier

def train_model_from_dataset(vectorizer, classifier, dataset_path, do_cross_validation=True):
    try:
        df = pd.read_csv(dataset_path, encoding='ISO-8859-9')
        df = df.dropna(subset=["label", "text"])
        print("df =", df)

        df['label'] = df['label'].map({'Ham': 0, 'Spam': 1})

        if df['label'].isnull().any():
            raise ValueError("Label column contains unknown classes besides 'Ham' and 'Spam'")

        X_raw = df["text"].values
        y = df["label"].values
        # print("X_raw =", X_raw)
        # print("y =", y)

        if do_cross_validation:
            print("Running Stratified K-Fold Cross-Validation...")
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            losses = []

            for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y), 1):
                vec = vectorizer.__class__(**vectorizer.get_params())
                clf = classifier.__class__(**classifier.get_params())

                X_train = vec.fit_transform(X_raw[train_idx])
                X_test = vec.transform(X_raw[test_idx])
                y_train, y_test = y[train_idx], y[test_idx]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test)  # Gerekli: log loss için olasılıklar

                acc = accuracy_score(y_test, y_pred)
                loss = log_loss(y_test, y_proba)

                scores.append(acc)
                losses.append(loss)

                print(f"Fold {fold} Accuracy: {acc:.4f} | Log Loss: {loss:.4f}")

            print(f"\nAverage Cross-Validation Accuracy: {np.mean(scores):.4f}")
            print(f"Average Cross-Validation Log Loss: {np.mean(losses):.4f}\n")

        # Final model training
        X = vectorizer.fit_transform(X_raw)
        classifier.fit(X, y)

        return vectorizer, classifier

    except Exception as e:
        print(f"Error training model from dataset: {e}")
        raise

def load_spam_model():
    dataset_path = "C:\\Users\\Administrator\\source\\repos\\Capstone\\Capstone\\spam_ham_dataset.csv"
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_files")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    model_path = os.path.join(model_dir, "spam_model.pkl")

    # Creates directory if not exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    try:
        if os.path.exists(vectorizer_path) and os.path.exists(model_path):
            vectorizer = joblib.load(vectorizer_path)
            classifier = joblib.load(model_path)
            return vectorizer, classifier
    except Exception as e:
        print(f"Error occured while model loading: {e}")

    # Create and train a model if model couldn't load
    print("Model is creating and Training...")
    vectorizer, classifier = create_basic_model()
    vectorizer, classifier = train_model_from_dataset(vectorizer, classifier, dataset_path)

    try:
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(classifier, model_path)
    except Exception as e:
        print(f"Trained model can't be saved: {e}")

    return vectorizer, classifier


# Extract IP from the Received header
def extract_ip_from_received(header):
    pattern = r'\[(\d+\.\d+\.\d+\.\d+)\]'
    match = re.search(pattern, header)
    return match.group(1) if match else None

# Parse email content
def parse_email(email_content):
    try:
        msg = BytesParser(policy=policy.default).parsebytes(email_content.encode())
    except:
        # Fallback if already bytes or other encoding issue
        try:
            msg = BytesParser(policy=policy.compat32).parsebytes(email_content)
        except:
            # If all else fails, try as string
            msg = email.message_from_string(email_content)
    
    received_headers = msg.get_all('Received')
    sender_ip = None
    if received_headers:
        last_received = received_headers[-1]
        sender_ip = extract_ip_from_received(last_received)
    
    subject = msg.get("Subject", "")
    
    # Better body extraction with HTML handling
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                try:
                    body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    body += part.get_payload(decode=True).decode('ISO-8859-1', errors='ignore')
            elif content_type == 'text/html':
                try:
                    html_content = part.get_payload(decode=True).decode(errors='ignore')
                    body += html2text(html_content)
                except:
                    pass
    else:
        content_type = msg.get_content_type()
        if content_type == 'text/html':
            try:
                html_content = msg.get_payload(decode=True).decode(errors='ignore')
                body += html2text(html_content)
            except:
                body += str(msg.get_payload())
        else:
            try:
                body += msg.get_payload(decode=True).decode(errors='ignore')
            except:
                body += str(msg.get_payload())
    
    # Extract attachments
    attachments = []
    for part in msg.walk():
        if part.get_content_disposition() == 'attachment' or part.get_filename():
            try:
                attachments.append(part.get_payload(decode=True))
            except:
                pass
    
    # Extract URLs using a more robust regex
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    urls = re.findall(url_pattern, body)
    if subject:
        urls.extend(re.findall(url_pattern, subject))
    
    return sender_ip, subject, body, attachments, urls

# Text preprocessing that preserves more signal
def preprocess_text(text):
    # Keep some punctuation that might be relevant for spam detection
    text = re.sub(r'[^\w\s!?.,$@#%]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Check IP reputation
def check_abuseipdb(ip):
    api_key = os.getenv('ABUSEIPDB_API_KEY')
    if not api_key:
        print("Warning: No AbuseIPDB API key found")
        return 0
    
    url = 'https://api.abuseipdb.com/api/v2/check'
    params = {'ipAddress': ip, 'maxAgeInDays': 90}
    headers = {'Key': api_key, 'Accept': 'application/json'}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            score = data.get('data', {}).get('abuseConfidenceScore', 0)
            # Scale score based on confidence
            if score > 80:
                return 3  # High risk
            elif score > 50:
                return 2  # Medium risk
            elif score > 20:
                return 1  # Low risk
            return 0
        return 0
    except Exception as e:
        print(f"AbuseIPDB error: {e}")
        return 0

# Replace the transformer-based model with our own simple model
def classify_spam_content(vectorizer, classifier, text):
    try:
        # Ensure we have text to analyze
        if not text or len(text.strip()) == 0:
            return 0, 0.0
        
        # Transform text using vectorizer
        X = vectorizer.transform([text])
        
        # Get prediction probability
        spam_prob = classifier.predict_proba(X)[0][1]
        
        # Assign scores based on probability
        if spam_prob > 0.9:
            score = 3  # High confidence spam
        elif spam_prob > 0.7:
            score = 2  # Medium confidence spam
        elif spam_prob > 0.4:
            score = 1  # Low confidence spam
        else:
            score = 0  # Likely ham
            
        return score, spam_prob
    except Exception as e:
        print(f"Classification error: {e}")
        return 0, 0.0

# Improved keyword checking
def check_keywords(text, keywords=SPAM_KEYWORDS):
    text_lower = text.lower()
    found_keywords = [kw for kw in keywords if kw in text_lower]
    
    # Score based on number of keywords found
    if len(found_keywords) >= 5:
        return 2, found_keywords
    elif len(found_keywords) > 1:
        return 1, found_keywords
    return 0, found_keywords

# Check VirusTotal for file hash
def check_virustotal_file(file_hash):
    api_key = os.getenv('VIRUSTOTAL_API_KEY')
    if not api_key:
        print("Warning: No VirusTotal API key found")
        return 0
    
    url = 'https://www.virustotal.com/vtapi/v2/file/report'
    params = {'apikey': api_key, 'resource': file_hash}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('response_code') == 1:  # File found in database
                positives = data.get('positives', 0)
                total = data.get('total', 0)
                
                # Score based on ratio of positives
                if total > 0:
                    ratio = positives / total
                    if ratio > 0.1:  # More than 10% engines detect as malicious
                        return 3
                    elif positives > 0:
                        return 2
            return 0
        return 0
    except Exception as e:
        print(f"VirusTotal file check error: {e}")
        return 0

# Check URL reputation
def check_url_reputation(url):
    api_key = os.getenv('VIRUSTOTAL_API_KEY')
    if not api_key:
        print("Warning: No VirusTotal API key found")
        
        # Fallback: Check for suspicious patterns in the URL
        suspicious_patterns = [
            r'\.(?:xyz|tk|ml|ga|cf|gq|pw)/', # Suspicious TLDs
            r'ip\d+', # IP addresses in subdomain
            r'(?:tiny|bit|goo)\.', # URL shorteners
            r'(?:bank|paypal|amazon|ebay|netflix).*\.(?!com|net|org)', # Brand impersonation
            r'\.(com|net)[-_.]' # Domain spoofing
        ]
        
        score = 0
        for pattern in suspicious_patterns:
            if re.search(pattern, url.lower()):
                score += 1
        
        return min(score, 2)  # Max score 2 for pattern matching
    
    url_endpoint = 'https://www.virustotal.com/vtapi/v2/url/report'
    params = {'apikey': api_key, 'resource': url}
    
    try:
        response = requests.get(url_endpoint, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('response_code') == 1:  # URL found in database
                positives = data.get('positives', 0)
                total = data.get('total', 0)
                
                # Score based on detections
                if total > 0:
                    ratio = positives / total
                    if ratio > 0.1:  # More than 10% engines flag as malicious
                        return 3
                    elif positives > 0:
                        return 1
            return 0
        return 0
    except Exception as e:
        print(f"URL reputation check error: {e}")
        return 0

# Main spam detection function
def is_spam_email(email_content):
    # Load vectorizer and model
    vectorizer, classifier = load_spam_model()
    
    # Parse email
    sender_ip, subject, body, attachments, urls = parse_email(email_content)
    
    # Initialize score and details
    score = 0
    details = {
        'scores': {},
        'explanations': {}
    }
    
    # Check sender IP
    if sender_ip:
        ip_score = check_abuseipdb(sender_ip)
        score += ip_score
        details['scores']['sender_ip'] = ip_score
        details['sender_ip'] = sender_ip
        
        if ip_score > 0:
            details['explanations']['ip'] = f"Sender IP {sender_ip} has suspicious reputation (score: {ip_score})"
    
    # Check subject
    if subject:
        subject_kw_score, subject_keywords = check_keywords(subject)
        score += subject_kw_score
        details['scores']['subject_keywords'] = subject_kw_score
        details['subject'] = subject
        
        if subject_kw_score > 0:
            details['explanations']['subject'] = f"Subject contains spam keywords: {', '.join(subject_keywords)}"
    
    # Check body with both keyword and ML approaches
    if body:
        # Preprocess without losing too much signal
        processed_body = preprocess_text(body)
        
        # Keyword check
        body_kw_score, body_keywords = check_keywords(processed_body)
        score += body_kw_score
        details['scores']['body_keywords'] = body_kw_score
        
        if body_kw_score > 0:
            details['explanations']['body_keywords'] = f"Email body contains spam keywords: {', '.join(body_keywords)}"
        
        # ML classification
        ml_score, ml_prob = classify_spam_content(vectorizer, classifier, processed_body)
        score += ml_score
        details['scores']['ml_classification'] = ml_score
        details['ml_spam_probability'] = ml_prob
        
        if ml_score > 0:
            details['explanations']['ml_model'] = f"ML model classified as spam with {ml_prob:.2%} confidence"
    
    # Check  URLs
    url_scores = []
    if urls:
        for url in urls:
            url_score = check_url_reputation(url)
            score += url_score
            url_scores.append((url, url_score))
            
            if url_score > 0:
                if 'urls' not in details['explanations']:
                    details['explanations']['urls'] = []
                details['explanations']['urls'].append(f"Suspicious URL: {url} (score: {url_score})")
    
    details['scores']['urls'] = url_scores
    
    # Check attachments
    attachment_scores = []
    if attachments:
        for i, attachment in enumerate(attachments):
            if attachment:
                attachment_hash = hashlib.sha256(attachment).hexdigest()
                attachment_score = check_virustotal_file(attachment_hash)
                score += attachment_score
                attachment_scores.append((f"attachment_{i+1}", attachment_score))
                
                if attachment_score > 0:
                    if 'attachments' not in details['explanations']:
                        details['explanations']['attachments'] = []
                    details['explanations']['attachments'].append(f"Suspicious attachment #{i+1} (score: {attachment_score})")
    
    details['scores']['attachments'] = attachment_scores
    
    # Calculate final score and verdict
    details['total_score'] = score
    
    # Threshold adjusted for better accuracy
    # Higher threshold as we're collecting more signals now
    is_spam = score >= 3
    
    return is_spam, details
