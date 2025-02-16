import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# NLTK stopwords yüklemesi
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Excel dosyasını oku
df = pd.read_excel('train_tweets.xlsx', header=None)
df.columns = ['comment', 'sentiment']  # varsayılan A ve B sütunları için isimlendirme

# Veriyi temizleme fonksiyonu
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # Sayıları kaldır
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    text = text.lower()  # Küçük harfe çevir
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Stop words kaldır
    return text

# Yorumları temizle
df['comment'] = df['comment'].apply(clean_text)

# Tokenizer kullanarak metinleri sayılara dönüştür
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['comment'])

# Metinleri sayılara dönüştür
X_seq = tokenizer.texts_to_sequences(df['comment'])

# Dizi uzunluklarını eşitlemek için padding ekle
max_length = max([len(x) for x in X_seq])  # en uzun cümlenin uzunluğuna göre padding
X_pad = pad_sequences(X_seq, maxlen=max_length)

# Tokenleştirilmiş verileri DataFrame'e dönüştür
tokenized_df = pd.DataFrame(X_pad)

# Tokenleştirilmiş veriyi CSV olarak kaydet
tokenized_df.to_csv('tokenized_comments.csv', index=False)

print("Tokenleştirilmiş veri 'tokenized_comments.csv' olarak kaydedildi.")
