from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import seaborn as sns
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from deep_translator import GoogleTranslator
import gensim
from gensim import corpora
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import networkx as nx
#from googletrans import Translator
extract=URLExtract()
#translator = Translator()

#blocked_words = {"message", "joined", "using", "link", "group", "community", "option:", "edited>","<this","added","please","settings","new","we're","pls","anyone","really","okayy","put","got","ask","we’re","person"}

blocked_words = {
    # WhatsApp system / metadata
    "message", "joined", "using", "link", "group", "community", "option:",
    "edited>", "<this", "<media", "omitted>", "admin", "settings",
    "add", "added", "remove", "removed",

    # Generic chat fillers
    "please", "pls", "plz", "ok", "okay", "okayy", "alright", "haan",
    "anyone", "everyone", "someone", "thing", "stuff", "really", "got",
    "put", "ask", "asked", "done", "like", "yeah", "yes", "no", "maybe",
    "thanks", "thankyou", "thank", "ty", "sorry",

    # Phone number / placeholder patterns
    "+91", "+1",

    # Conversation glue words (not topics)
    "we're", "we’re", "i’m", "im", "you’re", "ur", "bro", "dude",
    "guys", "yaar", "dear",

    # Frequency but no topic meaning
    "today", "tomorrow", "now", "then", "come", "go", "there",
    "here", "just", "can", "will", "should", "might",

    # URLs/text artifacts
    "http", "https", "www", "link.", "click", "photo",

    # Voting / poll noise
    "votes)", "option", "options" ,"get","deleted","evening"
}

hinglish_lexicon = {
    "mast": 2, "kadak": 2, "badiya": 2, "jhakas": 2, "solid": 2, "awesome": 2,
    "accha": 1, "shandar": 2, "kamal": 2, "jabardast": 3, "superb": 2,
    "nice": 1, "great": 2, "cool": 1, "perfect": 2, "legend": 3,
    "cute": 1, "smart": 1, "handsome": 1, "sundar": 1, "pyara": 1,
    "love": 2, "respect": 2, "happy": 1, "fun": 1, "funny": 1,
    "enjoy": 1, "dilkhush": 2, "amazing": 2, "op": 3, "fire": 2,
    "best": 2, "bhai": 1, "bhaiya": 1, "bhaukal": 2, "leader": 2,

    "sahi": 1, "theek": 1, "chill": 1, "nice": 1, "well": 1,
    "win": 2, "secured": 1, "achiever": 2, "hero": 2,

    "bekar": -1, "bakwas": -1, "ganda": -1, "maro": -1, "faltu": -1,
    "stupid": -2, "idiot": -2, "pagal": -1, "lodu": -2, "ullu": -2,
    "bewakoof": -2, "bhand": -1, "bewda": -1, "chutiya": -3, "chomu": -2,
    "madarchod": -3, "harami": -3, "bc": -2, "bkl": -2, "mc": -3,
    "lanat": -2, "ghatiya": -2, "kutta": -2, "kutti": -2, "nonsense": -2,
    "hate": -2, "bewkuf": -2, "nalayak": -2, "bewkoof": -2,

    "gussa": -1, "naraz": -1, "thappad": -2, "toddu": -2, "maar": -2,
    "fight": -1, "tension": -1, "sad": -1, "cry": -1, "ro": -1,

    "lol": 1, "lmao": 1, "rofl": 1, "noob": -1, "baklol": -1,
    "nalla": -2, "fattu": -2, "chillam": -1,

    "😂": 1, "🤣": 2, "😢": -1, "😭": -2, "😡": -2, "🤬": -3,
    "🤡": -1, "👌": 2, "👍": 2, "🔥": 3, "💀": -1, "❤️": 3,

    "shukriya": 1, "dhanyavaad": 1, "thanks": 1, "thankyou": 1,
    "badhiya": 2, "zindabad": 2, "proud": 2, "king": 3,

    "boring": -1, "bore": -1, "thak": -1, "nikaal": -1,
    "bacha": 1, "saviour": 2, "helpful": 1, "kind": 1,

    "scene": 0, "chal": 0, "chod": -2, "lag": -1, "lost": -1,
    "bakchodi": -1, "jhand": -2, "kat": -1, "dogla": -2, "chipku": -1,
    "dabba": -2, "faltu": -1, "sasta": -1
}

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def slang_sentiment(text):
    score = 0
    for word in text.lower().split():
        score += hinglish_lexicon.get(word, 0)
    return score


def fetch_stats(selected_user,df):

    # if(selected_user=='Overall'):
    #     #fetching number of messages
    #     num_messages=df.shape[0]
    #     #now fetching number of words
    #     words=[]
    #     for message in df['message']:
    #         words.extend(message.split())
    #     return num_messages,len(words)
    # else:
    #     new_df=df[df['user'] == selected_user]
    #     num_messages=new_df.shape[0]
    #     words=[]
    #     for message in new_df['message']:
    #         words.extend(message.split())
    #         return num_messages,len(words)

    #---------------Restructing code as above code is too bulky-------------#

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

        # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())


        # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
        
        #fetch number of links shared
    links=[]
    for message in df['message']:
            links.extend(extract.find_urls(message))


    return num_messages,len(words),num_media_messages,len(links)


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df


#wordcloud

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

#emoji analysis


def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df


def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


def translate_to_english(text):
    try:
        result = translator.translate(text, src="auto", dest="en")
        return result.text
    except:
        return text

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("sentiment_vectorizer.pkl", "rb"))


def ml_sentiment(selected_user, df):
    temp = df
    if selected_user != "Overall":
        temp = temp[temp['user'] == selected_user]

    translated = temp['message'].apply(translate_to_english)
    X = vectorizer.transform(translated)

    preds = model.predict(X)
    #temp['sentiment'] = preds
    temp['sentiment'] = preds
    temp['slang_score'] = temp['message'].apply(slang_sentiment)

    temp.loc[temp['slang_score'] <= -2, 'sentiment'] = 0  # Negative
    temp.loc[temp['slang_score'] >= 2, 'sentiment'] = 1  # Positive

    #result = temp['sentiment'].value_counts().replace({0: "Negative", 1: "Positive"})
    result = temp['sentiment'].value_counts()
    result = result.rename(index={0: "Negative", 1: "Positive"})
    return result

    return result


svm_model = pickle.load(open("svm_sentiment_model.pkl","rb"))
svm_vectorizer = pickle.load(open("svm_vectorizer.pkl","rb"))

def svm_sentiment(selected_user, df):
    temp = df
    if selected_user != "Overall":
        temp = temp[temp['user'] == selected_user]

    translated = temp['message'].apply(translate_to_english)
    X = vectorizer.transform(translated)
    preds = svm_model.predict(X)
    #temp['sentiment'] = preds
    temp['sentiment'] = preds
    temp['slang_score'] = temp['message'].apply(slang_sentiment)

    temp.loc[temp['slang_score'] <= -2, 'sentiment'] = 0
    temp.loc[temp['slang_score'] >= 2, 'sentiment'] = 1

    #return temp['sentiment'].value_counts().replace({0:"Negative",1:"Positive"})
    result = temp['sentiment'].value_counts()
    result = result.rename(index={0: "Negative", 1: "Positive"})
    return result


bert_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
bert_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def bert_predict_sentiment(text):
    tokens = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    output = bert_model(**tokens)
    scores = output.logits.softmax(dim=1).detach().numpy()
    label = scores.argmax(axis=1)[0] + 1  # 1-5 stars
    return label


def bert_sentiment(selected_user, df):
    temp = df
    if selected_user != "Overall":
        temp = temp[temp['user'] == selected_user]

    messages = temp['message'].tolist()
    batch_size = 32  # adjust if needed

    all_preds = []

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]

        tokens = bert_tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = bert_model(**tokens)

        scores = outputs.logits.softmax(dim=1).detach().numpy()
        labels = np.argmax(scores, axis=1) + 1
        all_preds.extend(labels)

    sentiment_map = {
        1: "Very Negative",
        2: "Negative",
        3: "Neutral",
        4: "Positive",
        5: "Very Positive"
    }

    temp['sentiment'] = [sentiment_map[l] for l in all_preds]

    return temp['sentiment'].value_counts()


def sentiment_timeline(selected_user, df, model_choice):
    # Choose model
    if model_choice == "Logistic Regression":
        df['sentiment'] = df['message'].apply(lambda x: model.predict(vectorizer.transform([x]))[0])
        df['sentiment'] = df['sentiment'].replace({1: "Positive", 0: "Negative"})

    elif model_choice == "SVM":
        df['sentiment'] = df['message'].apply(lambda x: svm_model.predict(svm_vectorizer.transform([x]))[0])
        df['sentiment'] = df['sentiment'].replace({1: "Positive", 0: "Negative"})

    elif model_choice == "BERT":
        df['sentiment'] = df['message'].apply(bert_predict_sentiment)
        df['sentiment'] = df['sentiment'].map({
            1: "Very Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Very Positive"
        })

    # Convert labels to scores
    sentiment_score = {
        "Very Negative": -2,
        "Negative": -1,
        "Neutral": 0,
        "Positive": 1,
        "Very Positive": 2
    }

    df['sentiment_score'] = df['sentiment'].map(sentiment_score)

    # Aggregate daily sentiment
    mood_timeline = df.groupby('only_date')['sentiment_score'].mean().reset_index()

    return mood_timeline


def topic_modeling(selected_user, df, num_topics=5):

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    messages = df['message'].tolist()

    # Preprocessing
    stop_words = set(stopwords.words('english'))
    stop_words.update(["<media", "omitted>"])  # add custom removals

    texts = []
    # for msg in messages:
    #     words = [w.lower() for w in msg.split() if w.lower() not in stop_words and len(w) > 2]
    #     texts.append(words)
    for msg in messages:
        # remove phone numbers +91xxxx...
        msg = re.sub(r'\+?\d[\d -]{7,}\d', '', msg)
        words = []
        for w in msg.lower().split():
            if w in blocked_words:
                continue
            if w.startswith("<") and w.endswith(">"):
                continue
            if w not in stop_words and len(w) > 2:
                words.append(w)
        texts.append(words)


    # Create dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # LDA Model
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10
    )

    topics = lda_model.print_topics(num_words=6)

    return topics

# ---------------- TOXICITY DETECTION (INNOVATION) ---------------- #

tox_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
tox_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

tox_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def toxicity_analysis(selected_user, df):
    temp = df.copy()

    if selected_user != "Overall":
        temp = temp[temp['user'] == selected_user]

    messages = temp['message'].tolist()

    if len(messages) == 0:
        return pd.DataFrame(), pd.Series(dtype=float)

    all_rows = []
    batch_size = 32

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        inputs = tox_tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = tox_model(**inputs)

        # Multi-label probabilities
        scores = torch.sigmoid(outputs.logits).cpu().numpy()

        for msg, score_vec in zip(batch, scores):
            row = {"message": msg}
            for label, s in zip(tox_labels, score_vec):
                row[label] = float(s)
            all_rows.append(row)

    tox_df = pd.DataFrame(all_rows)
    tox_df["max_toxic"] = tox_df[tox_labels].max(axis=1)

    # Overall toxicity profile (mean per category)
    summary = tox_df[tox_labels].mean().sort_values(ascending=False)

    return tox_df, summary


# ---------------- SOCIAL NETWORK GRAPH (INNOVATION) ---------------- #

def build_conversation_graph(df):
    """
    Build a directed graph where an edge A -> B means
    after A's message, B replied next.
    Edge weight = number of such transitions.
    """
    G = nx.DiGraph()

    # We assume df is in chronological order
    users_series = df['user'].tolist()

    for i in range(len(users_series) - 1):
        u = users_series[i]
        v = users_series[i + 1]

        # ignore system messages
        if u == "group_notification" or v == "group_notification":
            continue

        if u == v:
            continue

        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)

    return G




