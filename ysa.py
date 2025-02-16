import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from tensorflow.keras.utils import plot_model  # plot_model eklendi
from sklearn.metrics import classification_report


# NLTK stopwords yüklemesi
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Excel dosyasını oku
df = pd.read_excel('train_tweets.xlsx', header=None)
df.columns = ['comment', 'sentiment']  # Varsayılan A ve B sütunları için isimlendirme

# Veriyi temizleme fonksiyonu
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # Sayıları kaldır
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    text = text.lower()  # Küçük harfe çevir
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Stop words kaldır
    return text

# Yorumları temizle
df['comment'] = df['comment'].apply(clean_text)

# Sentiment etiketi sayısal hale getirilmesi
label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

# Yorumları ve etiketleri ayır
X = df['comment'].values
y = df['sentiment'].values

# Eğitim ve test verilerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenizer kullanarak metinleri sayılara dönüştür
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Dizi uzunluklarını eşitlemek için padding ekle
max_length = max([len(x) for x in X_train_seq])  # En uzun cümlenin uzunluğuna göre padding
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

# Word2Vec ile kelime vektörleri oluşturma
model_w2v = Word2Vec(sentences=[text.split() for text in X_train], vector_size=100, window=5, min_count=1, workers=4)

# Kelimeleri vektörlere dönüştürme
def get_word_vectors(text):
    vectors = []
    for word in text.split():
        if word in model_w2v.wv:
            vectors.append(model_w2v.wv[word])
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

# Eğitim verilerini vektörleştir
X_train_w2v = np.array([get_word_vectors(text) for text in X_train])
X_test_w2v = np.array([get_word_vectors(text) for text in X_test])

# LSTM tabanlı bir model kur
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=max_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))  # 3 sınıf (pozitif, negatif, nötr)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğit (epoch sayısı 10 olarak güncellendi)
model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_data=(X_test_pad, y_test))

# Modelin topolojisini kaydet
plot_model(model, to_file='model_topology.png', show_shapes=True, show_layer_names=True)

# Test verisi üzerinde tahmin yap
y_pred = np.argmax(model.predict(X_test_pad), axis=1)

# Sonuçları değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Verisi Doğruluğu: {accuracy:.4f}')


# Yeni yorum dosyasını oku
yorumlar_df = pd.read_csv('yorumlar.csv')

# Yorumları modelle uyumlu hale getirmek için temizleme işlemi
yorumlar_df['Yorum'] = yorumlar_df['Yorum'].apply(clean_text)  # Daha önce tanımladığımız clean_text fonksiyonu

# Yeni yorumları tokenize et ve padding işlemini uygula
yorumlar_seq = tokenizer.texts_to_sequences(yorumlar_df['Yorum'])
yorumlar_pad = pad_sequences(yorumlar_seq, maxlen=max_length)

# Duygu tahmini yap
yorumlar_df['Duygu'] = np.argmax(model.predict(yorumlar_pad), axis=1)

# Duygu tahminlerini etiket olarak geri dönüştür (sayısal değerleri etikete çevir)
yorumlar_df['Duygu'] = label_encoder.inverse_transform(yorumlar_df['Duygu'])

# Sonuçları yeni bir CSV dosyasına kaydet
yorumlar_df.to_csv('yorumlar_with_sentiment.csv', index=False)

print("Duygu etiketi tahmin edilip yorumlar_with_sentiment.csv dosyasına kaydedildi.")

# Yeni yorum dosyasını oku
yorumlar_df = pd.read_csv('yorumlar.csv')

# Yorumları modelle uyumlu hale getirmek için temizleme işlemi
yorumlar_df['Yorum'] = yorumlar_df['Yorum'].apply(clean_text)  # Daha önce tanımladığımız clean_text fonksiyonu

# Yeni yorumları tokenize et ve padding işlemini uygula
yorumlar_seq = tokenizer.texts_to_sequences(yorumlar_df['Yorum'])
yorumlar_pad = pad_sequences(yorumlar_seq, maxlen=max_length)

# Duygu tahmini yap
yorumlar_df['Duygu'] = np.argmax(model.predict(yorumlar_pad), axis=1)

# Duygu tahminlerini etiket olarak geri dönüştür (sayısal değerleri etikete çevir)
yorumlar_df['Duygu'] = label_encoder.inverse_transform(yorumlar_df['Duygu'])

# Sonuçları yeni bir CSV dosyasına kaydet
yorumlar_df.to_csv('yorumlar_with_sentiment.csv', index=False)

# Yüzde dağılımını hesaplama
sentiment_counts = yorumlar_df['Duygu'].value_counts(normalize=True) * 100

# Sonuçları terminalde göster
print("\nDuygu Dağılımı:")
for sentiment, percentage in sentiment_counts.items():
    print(f"{sentiment}: %{percentage:.2f}")

# Gerçek ve tahmin edilen etiketler (test seti üzerinden doğruluk analizi)
y_true = y_test
y_pred = np.argmax(model.predict(X_test_pad), axis=1)

# Her sınıfın doğruluğunu hesapla ve göster
report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)

print("\nDuygu Doğruluk Raporu:")
print(report)

print("\nDuygu etiketi tahmin edilip yorumlar_with_sentiment.csv dosyasına kaydedildi.")

