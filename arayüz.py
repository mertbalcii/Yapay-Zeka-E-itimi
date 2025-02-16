from googleapiclient.discovery import build
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk

# Sabit API anahtarı ve video ID
API_KEY = "AIzaSyDtNEeHurTqas9Ka49Ip6tlICUdse2TfBw"
VIDEO_ID = "bmxM1ypnI0Q"  # Otomatik olarak doldurulacak video ID

def yorumlari_cek(video_id):
    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)
        yorumlar = []
        next_page_token = None

        while True:
            cevap = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            ).execute()

            for item in cevap['items']:
                yorum = item['snippet']['topLevelComment']['snippet']
                yorumlar.append({
                    "Yazan": yorum['authorDisplayName'],
                    "Yorum": yorum['textDisplay'],
                    "Beğeni Sayısı": yorum['likeCount'],
                    "Tarih": yorum['publishedAt']  # Yorumun yayınlandığı tarih
                })

            next_page_token = cevap.get('nextPageToken')
            if not next_page_token:
                break

        return yorumlar
    except Exception as e:
        messagebox.showerror("Hata", f"Yorumları çekerken bir hata oluştu:\n{e}")
        return None

def kaydet(yorumlar):
    dosya_adi = "yorumlar.csv"
    try:
        df = pd.DataFrame(yorumlar)
        df.to_csv(dosya_adi, index=False, encoding="utf-8")
        messagebox.showinfo("Başarılı", f"Yorumlar başarıyla {dosya_adi} dosyasına kaydedildi!")
    except Exception as e:
        messagebox.showerror("Hata", f"Dosya kaydedilirken bir hata oluştu:\n{e}")

def tabloyu_goster(yorumlar):
    # Mevcut tablonun verilerini temizle
    for row in tree.get_children():
        tree.delete(row)
    
    # Yorumları tabloya ekle
    for yorum in yorumlar:
        tree.insert("", tk.END, values=(yorum['Yazan'], yorum['Yorum'], yorum['Beğeni Sayısı'], yorum['Tarih']))

def baslat():
    video_id = video_id_giris.get().strip()

    if not video_id:
        messagebox.showerror("Hata", "Video ID eksik!")
        return

    yorumlar = yorumlari_cek(video_id)
    if yorumlar:
        kaydet(yorumlar)
        tabloyu_goster(yorumlar)

# Arayüz oluşturma
pencere = tk.Tk()
pencere.title("YouTube Yorum Çekme Aracı")

tk.Label(pencere, text="Video ID:").grid(row=0, column=0, padx=10, pady=10)

# Video ID otomatik olarak dolduruluyor
video_id_giris = tk.Entry(pencere, width=40)
video_id_giris.insert(0, VIDEO_ID)  # Otomatik doldurma
video_id_giris.grid(row=0, column=1, padx=10, pady=10)

# Yorumları gösteren tablo (Treeview)
tree = ttk.Treeview(pencere, columns=("Yazan", "Yorum", "Beğeni Sayısı", "Tarih"), show="headings", height=15)
tree.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Sütun başlıklarını ekle
tree.heading("Yazan", text="Yazan")
tree.heading("Yorum", text="Yorum")
tree.heading("Beğeni Sayısı", text="Beğeni Sayısı")
tree.heading("Tarih", text="Tarih")

# Sütun genişliklerini ayarla
tree.column("Yazan", width=150)
tree.column("Yorum", width=1000)
tree.column("Beğeni Sayısı", width=50)
tree.column("Tarih", width=150)

# Buton
tk.Button(pencere, text="Yorumları Çek ve Kaydet", command=baslat).grid(row=2, column=0, columnspan=2, pady=10)

pencere.mainloop()
