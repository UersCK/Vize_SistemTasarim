import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                            QComboBox, QMessageBox, QFrame, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
import numpy as np
from ultralytics import YOLO
import time
import math

# SporHareketAnalizi sınıfı için
class SporHareketAnalizi:
    def __init__(self):
        self.model = YOLO("yolov8n-pose.pt")
        self.squat_sayaci = 0
        self.sinav_sayaci = 0
        self.kol_kaldirma_sayaci = 0
        self.aktif_mod = ""  # Başlangıçta boş - kullanıcının seçmesi gerekir
        
        # Her hareket için ayrı durum takibi
        self.squat_son_durum = "baslangic"
        self.sinav_son_durum = "baslangic"
        self.kol_son_durum = "baslangic"
        
        # Durum zamanları
        self.squat_son_durum_zamani = time.time()
        self.sinav_son_durum_zamani = time.time()
        self.kol_son_durum_zamani = time.time()
    
    def hareket_analiz(self, keypoints, frame_height):
        # Keypoints yoksa erken dön
        if keypoints is None or len(keypoints) < 17:
            return ["Vücut tespiti başarısız", False]
        
        # Eğer aktif mod seçilmemişse
        if not self.aktif_mod:
            return ["Lütfen önce bir egzersiz modu seçin!", False]
        
        aciklama = ""
        hareket_tamamlandi = False
        
        if self.aktif_mod == "squat":
            return self.squat_analiz(keypoints, frame_height)
        elif self.aktif_mod == "sinav":
            return self.sinav_analiz(keypoints, frame_height)
        elif self.aktif_mod == "kol_kaldirma":
            return self.kol_kaldirma_analiz(keypoints, frame_height)
        
        return [aciklama, hareket_tamamlandi]
        
    def squat_analiz(self, keypoints, frame_height):
        try:
            # Önemli noktaları al
            # 11: sol kalça, 12: sağ kalça, 13: sol diz, 14: sağ diz, 15: sol ayak, 16: sağ ayak
            # 5: sol omuz, 6: sağ omuz (vücut pozisyonunu kontrol etmek için)
            sol_kalca = keypoints[11][:2]
            sag_kalca = keypoints[12][:2]
            sol_diz = keypoints[13][:2]
            sag_diz = keypoints[14][:2]
            sol_ayak = keypoints[15][:2]
            sag_ayak = keypoints[16][:2]
            sol_omuz = keypoints[5][:2]
            sag_omuz = keypoints[6][:2]
            
            # Güven değerleri kontrolü
            if (keypoints[11][2] < 0.5 or keypoints[12][2] < 0.5 or
                keypoints[13][2] < 0.5 or keypoints[14][2] < 0.5 or
                keypoints[15][2] < 0.5 or keypoints[16][2] < 0.5 or
                keypoints[5][2] < 0.5 or keypoints[6][2] < 0.5):
                return ["Vücut noktaları net değil, pozisyonunu düzelt", False]
            
            # Orta noktaları hesapla
            kalca_orta = [(sol_kalca[0] + sag_kalca[0]) / 2, (sol_kalca[1] + sag_kalca[1]) / 2]
            diz_orta = [(sol_diz[0] + sag_diz[0]) / 2, (sol_diz[1] + sag_diz[1]) / 2]
            ayak_orta = [(sol_ayak[0] + sag_ayak[0]) / 2, (sol_ayak[1] + sag_ayak[1]) / 2]
            omuz_orta = [(sol_omuz[0] + sag_omuz[0]) / 2, (sol_omuz[1] + sag_omuz[1]) / 2]
            
            # Kalça-diz-ayak açısını hesapla
            kalca_diz_ayak_acisi = self.aci_hesapla(kalca_orta, diz_orta, ayak_orta)
            
            # Debug için log
            print(f"[Squat] Açı: {kalca_diz_ayak_acisi:.1f}°")
            
            # Dikey açıyı hesapla (kalça-diz-y ekseni)
            dikey_aci = self.dikey_aci_hesapla(kalca_orta, diz_orta)
            
            # Vücut dik duruyor mu kontrol et (şınav pozisyonu olmamalı)
            omuz_kalca_aci = self.dikey_aci_hesapla(omuz_orta, kalca_orta)
            vucut_dik_mi = omuz_kalca_aci > 45  # Omuz-kalça açısı dikeye yakın olmalı (şınavda yatay olur)
            
            # Squat için geçersiz pozisyon kontrolü - eller yerde ise şınav yapıyor olabilir
            eller_yerde_mi = (keypoints[9][1] > frame_height * 0.7 or keypoints[10][1] > frame_height * 0.7)
            
            # Şınav pozisyonunda ise squat sayma
            if eller_yerde_mi and not vucut_dik_mi:
                return ["Bu şınav pozisyonu, squat için ayakta durun", False]
            
            # Squat için eşik değerleri
            squat_acisi_esik = 120  # 120 dereceden küçük açılar squat pozisyonu
            dik_durum_esik = 160  # 160 dereceden büyük açılar dik duruş
            
            # Durum takibi
            suanki_zaman = time.time()
            
            # Dik durumdan squat pozisyonuna geçiş
            if self.squat_son_durum == "baslangic" or self.squat_son_durum == "dik":
                if kalca_diz_ayak_acisi < squat_acisi_esik:
                    self.squat_son_durum = "squat"
                    self.squat_son_durum_zamani = suanki_zaman
                    return ["Squat pozisyonunda, biraz bekle", False]
                else:
                    return ["Dik dur ve squat yapmak için çömel", False]
            
            # Squat pozisyonundan dik duruma geçiş
            elif self.squat_son_durum == "squat":
                # Squat pozisyonunda en az 0.5 saniye kalmak gerekiyor
                if suanki_zaman - self.squat_son_durum_zamani > 0.5:
                    if kalca_diz_ayak_acisi > dik_durum_esik:
                        self.squat_son_durum = "dik"
                        self.squat_son_durum_zamani = suanki_zaman
                        self.squat_sayaci += 1
                        return [f"Squat tamamlandı! Toplam: {self.squat_sayaci}", True]
                    else:
                        return ["Squat pozisyonundan dik duruma geç", False]
                else:
                    return ["Squat pozisyonunda tut...", False]
            
            return ["Squat için hazır", False]
        
        except Exception as e:
            return [f"Squat analiz hatası: {str(e)}", False]
    
    def sinav_analiz(self, keypoints, frame_height):
        try:
            # Gerekli noktaları al
            sol_omuz = keypoints[5][:2]
            sag_omuz = keypoints[6][:2]
            sol_dirsek = keypoints[7][:2]
            sag_dirsek = keypoints[8][:2]
            sol_bilek = keypoints[9][:2]
            sag_bilek = keypoints[10][:2]
            
            # Güven değerleri kontrolü (indeks 2 güven değerini içerir)
            if (keypoints[5][2] < 0.5 or keypoints[6][2] < 0.5 or 
                keypoints[7][2] < 0.5 or keypoints[8][2] < 0.5 or 
                keypoints[9][2] < 0.5 or keypoints[10][2] < 0.5):
                return ["Vücut noktaları net değil, pozisyonunu düzelt", False]

            # Açılar hesapla
            sol_aci = self.aci_hesapla(sol_omuz, sol_dirsek, sol_bilek)
            sag_aci = self.aci_hesapla(sag_omuz, sag_dirsek, sag_bilek)
            omuz_dirsek_bilek_acisi = (sol_aci + sag_aci) / 2

            # Debug için detaylı log
            print(f"[ŞINAV DEBUG] Açılar → Sol: {sol_aci:.1f}, Sağ: {sag_aci:.1f}, Ortalama: {omuz_dirsek_bilek_acisi:.1f}")
            print(f"[ŞINAV DEBUG] Sol bilek Y: {sol_bilek[1]:.1f}, Sağ bilek Y: {sag_bilek[1]:.1f}, Frame Yüksekliği: {frame_height}")
            
            # Eller yere yakın mı? (daha esnek eşik)
            eller_yerde_mi = (sol_bilek[1] > frame_height * 0.6 and sag_bilek[1] > frame_height * 0.6)
            
            # Eşikler - biraz daha geniş aralık
            asagi_esik = 130  # 130 derece ve altında aşağıda
            yukari_esik = 130  # 130 derece ve üstünde yukarıda (eşik değeri daha da düşürüldü)
            
            # Hangi durumun kontrol edileceğini debug çıktısında göster
            print(f"[ŞINAV DEBUG] Eller yerde mi?: {eller_yerde_mi} [Durum: {self.sinav_son_durum}]")
            
            # Şu anki zamanı al
            suanki_zaman = time.time()
            durum_suresi = suanki_zaman - self.sinav_son_durum_zamani
            print(f"[ŞINAV DEBUG] Mevcut durumda geçen süre: {durum_suresi:.1f} saniye")

            # Başlangıç veya yukarıdaysa
            if self.sinav_son_durum == "baslangic" or self.sinav_son_durum == "yukari":
                if eller_yerde_mi and omuz_dirsek_bilek_acisi < asagi_esik:
                    self.sinav_son_durum = "asagi"
                    self.sinav_son_durum_zamani = suanki_zaman
                    print(f"[ŞINAV DEBUG] DURUM DEĞİŞTİ: yukari -> asagi (Açı: {omuz_dirsek_bilek_acisi:.1f})")
                    return [f"Aşağı pozisyon algılandı (Açı: {int(omuz_dirsek_bilek_acisi)}°), yukarı çık", False]
                else:
                    return [f"Aşağı inmelisin | Açın: {int(omuz_dirsek_bilek_acisi)}° [Durum: {self.sinav_son_durum}]", False]

            # Aşağıdaysa
            elif self.sinav_son_durum == "asagi":
                # En az bir saniye aşağıda kalma kontrolü - yanlış algılamaları önlemek için
                if durum_suresi < 0.5:
                    return [f"Şınav pozisyonunda tut... | Açı: {int(omuz_dirsek_bilek_acisi)}°", False]
                
                if omuz_dirsek_bilek_acisi > yukari_esik:
                    self.sinav_sayaci += 1
                    self.sinav_son_durum = "yukari"
                    self.sinav_son_durum_zamani = suanki_zaman
                    print(f"[ŞINAV DEBUG] DURUM DEĞİŞTİ: asagi -> yukari (Açı: {omuz_dirsek_bilek_acisi:.1f})")
                    return [f"Şınav sayıldı! Toplam: {self.sinav_sayaci}", True]
                else:
                    return [f"Yukarı çıkmalısın | Açın: {int(omuz_dirsek_bilek_acisi)}° [Durum: {self.sinav_son_durum}]", False]

            return [f"Şınav hareketi hazır - pozisyon al [Durum: {self.sinav_son_durum}]", False]

        except Exception as e:
            return [f"Hata oluştu: {str(e)}", False]
    
    def kol_kaldirma_analiz(self, keypoints, frame_height):
        try:
            # Önemli noktaları al
            # 5: sol omuz, 6: sağ omuz, 7: sol dirsek, 8: sağ dirsek, 9: sol bilek, 10: sağ bilek
            sol_omuz = keypoints[5][:2]
            sag_omuz = keypoints[6][:2]
            sol_dirsek = keypoints[7][:2]
            sag_dirsek = keypoints[8][:2]
            sol_bilek = keypoints[9][:2]
            sag_bilek = keypoints[10][:2]
            
            # Güven değerleri kontrolü
            if (keypoints[5][2] < 0.5 or keypoints[6][2] < 0.5 or
                keypoints[7][2] < 0.5 or keypoints[8][2] < 0.5 or
                keypoints[9][2] < 0.5 or keypoints[10][2] < 0.5):
                return ["Vücut noktaları net değil, pozisyonunu düzelt", False]
            
            # Dirsek-omuz-kalça açılarını hesapla
            sol_acı = self.aci_hesapla(sol_bilek, sol_dirsek, sol_omuz)
            sag_acı = self.aci_hesapla(sag_bilek, sag_dirsek, sag_omuz)
            
            # Ortalama açı
            avg_aci = (sol_acı + sag_acı) / 2
            
            # Debug için log
            print(f"[Kol] Açı: {avg_aci:.1f}°")
            
            # Kol kaldırma için eşik değerleri
            asagi_esik = 80   # Kollar aşağıda
            yukari_esik = 160  # Kollar yukarıda
            
            # Durum takibi
            suanki_zaman = time.time()
            
            # Aşağıdan yukarı geçiş
            if self.kol_son_durum == "baslangic" or self.kol_son_durum == "asagi":
                if avg_aci > yukari_esik:
                    self.kol_son_durum = "yukari"
                    self.kol_son_durum_zamani = suanki_zaman
                    return ["Yukarı pozisyonda, biraz bekle", False]
                else:
                    return ["Kolları yukarı kaldır", False]
            
            # Yukarıdan aşağı geçiş
            elif self.kol_son_durum == "yukari":
                # Yukarı pozisyonunda en az 0.2 saniye kalmak gerekiyor
                if suanki_zaman - self.kol_son_durum_zamani > 0.2:
                    if avg_aci < asagi_esik:
                        self.kol_son_durum = "asagi"
                        self.kol_son_durum_zamani = suanki_zaman
                        self.kol_kaldirma_sayaci += 1
                        return [f"Kol kaldırma tamamlandı! Toplam: {self.kol_kaldirma_sayaci}", True]
                    else:
                        return ["Kolları aşağı indir", False]
                else:
                    return ["Yukarı pozisyonda tut...", False]
            
            return ["Kol kaldırma için hazır", False]
        
        except Exception as e:
            return [f"Kol kaldırma analiz hatası: {str(e)}", False]
    
    def hareket_rehberlik(self, frame, keypoints, analiz_sonucu):
        # Rehberlik ve açıklamaları ekrandan kaldırıyoruz
        # Artık ekranın altında kırmızı metin gösterilmeyecek
        pass
        
    def aci_hesapla(self, p1, p2, p3):
        # Üç nokta arasındaki açıyı hesaplar (p2 merkez nokta)
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        # Radyandan dereceye çevir
        angle_deg = np.degrees(angle)
        
        return angle_deg
        
    def dikey_aci_hesapla(self, p1, p2):
        # İki nokta ile dikey eksen arasındaki açıyı hesaplar
        a = np.array(p1)
        b = np.array(p2)
        
        # Dikey vektör (y ekseni)
        vertical = np.array([0, 1])
        
        # İki nokta arasındaki vektör
        vec = b - a
        
        # Açıyı hesapla
        cosine_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        # Radyandan dereceye çevir
        angle_deg = np.degrees(angle)
        
        return angle_deg

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, dict)
    finished_signal = pyqtSignal(dict)

    def __init__(self, mode="webcam", video_path=None):
        super().__init__()
        self.mode = mode
        self.video_path = video_path
        self.running = True
        self.paused = False
        self.stats = {
            "squat_sayisi": 0,
            "sinav_sayisi": 0,
            "kol_kaldirma_sayisi": 0,
            "aktif_mod": "",  # Başlangıçta boş mod
            "aciklamalar": "Lütfen bir egzersiz türü seçin"
        }
        
        # SporHareketAnalizi sınıfını başlat
        self.analiz = SporHareketAnalizi()
        
    def change_mode(self, mod):
        self.stats["aktif_mod"] = mod
        self.analiz.aktif_mod = mod
        
        # Mod değiştiğinde ilgili son durumu sıfırla
        if mod == "squat":
            self.analiz.squat_son_durum = "baslangic"
            self.analiz.squat_son_durum_zamani = time.time()
        elif mod == "sinav":
            self.analiz.sinav_son_durum = "baslangic"
            self.analiz.sinav_son_durum_zamani = time.time()
        elif mod == "kol_kaldirma":
            self.analiz.kol_son_durum = "baslangic"
            self.analiz.kol_son_durum_zamani = time.time()
            
        print(f"Mod değiştirildi: {mod}")
        
    def reset_counter(self):
        if self.stats["aktif_mod"] == "squat":
            self.analiz.squat_sayaci = 0
        elif self.stats["aktif_mod"] == "sinav":
            self.analiz.sinav_sayaci = 0
        elif self.stats["aktif_mod"] == "kol_kaldirma":
            self.analiz.kol_kaldirma_sayaci = 0
            
    def toggle_pause(self):
        self.paused = not self.paused
        
    def stop(self):
        self.running = False
        self.wait()
    
    def run(self):
        if self.mode == "webcam":
            # Webcam'i aç
            cap = cv2.VideoCapture(0)
        else:
            # Video dosyasını aç
            cap = cv2.VideoCapture(self.video_path)
            
        if not cap.isOpened():
            self.finished_signal.emit({"error": "Kamera veya video açılamadı."})
            return
            
        # FPS hesaplama için değişkenler
        prev_time = 0
        total_frames = 0
        
        if self.mode == "video":
            # Video dosyasının toplam kare sayısını al
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while self.running:
            if not self.paused:
                # Bir kare al
                success, frame = cap.read()
                if not success:
                    if self.mode == "video":
                        # Video dosyası sonuna gelindi
                        self.stats["aciklamalar"] = "Video tamamlandı."
                        self.finished_signal.emit(self.stats)
                    else:
                        self.stats["aciklamalar"] = "Kamera hata verdi."
                        self.finished_signal.emit(self.stats)
                    break
                
                # FPS hesapla
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time
                
                # Modeli kullanarak pozu tespit et
                results = self.analiz.model(frame)
                
                # İşlenmiş kareyi al
                annotated_frame = results[0].plot()
                
                # Keypoints verilerini al
                try:
                    keypoints_data = results[0].keypoints.data
                    if len(keypoints_data) > 0:
                        keypoints = keypoints_data[0].cpu().numpy()
                        
                        # Hareket analizi yap
                        analiz_sonucu = self.analiz.hareket_analiz(keypoints, frame.shape[0])
                        
                        # Rehberlik bilgisi ekle
                        self.analiz.hareket_rehberlik(annotated_frame, keypoints, analiz_sonucu)
                        
                        # İstatistikleri güncelle
                        self.stats["squat_sayisi"] = self.analiz.squat_sayaci
                        self.stats["sinav_sayisi"] = self.analiz.sinav_sayaci
                        self.stats["kol_kaldirma_sayisi"] = self.analiz.kol_kaldirma_sayaci
                        self.stats["aciklamalar"] = analiz_sonucu[0]
                        
                except Exception as e:
                    print(f"Analiz hatası: {e}")
                
                # Kırmızıyla işaretlenen ekran bilgilerini kaldırıyoruz
                # Mod bilgisi, sayaç, FPS ve ilerleme bilgileri artık görüntülenmeyecek
                
                # Duraklatma bilgisi
                if self.paused:
                    cv2.putText(annotated_frame, "DURAKLATILDI", (frame.shape[1] - 200, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # İşlenmiş kareyi sinyal olarak gönder
                self.change_pixmap_signal.emit(annotated_frame, self.stats)
            
            # Bekleme süresi
            time.sleep(0.03)
        
        # Kaynakları serbest bırak
        cap.release()


class SporHareketAnaliziApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spor Hareket Analizi")
        self.setGeometry(100, 100, 1000, 700)
        self.setup_ui()
        
        # Video işleme thread'i
        self.thread = None
        
        # Çalıştıktan sonra yüklenmesini beklemek için timer
        self.timer = QTimer()
        self.timer.singleShot(500, self.show_ready_message)
        
    def show_ready_message(self):
        QMessageBox.information(self, "Hazır", "Spor Hareket Analizi uygulaması başlatıldı. Webcam veya video dosyası seçerek başlayabilirsiniz.")
        
    def setup_ui(self):
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Video görüntüleme alanı
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(640, 480)
        self.main_layout.addWidget(self.video_label)
        
        # İstatistik ve bilgi alanı
        self.stats_frame = QFrame()
        self.stats_frame.setFrameShape(QFrame.StyledPanel)
        self.stats_frame.setStyleSheet("background-color: #f0f0f0; padding: 10px;")
        self.stats_layout = QHBoxLayout(self.stats_frame)
        
        # Squat sayacı
        self.squat_label = QLabel("Squat Sayısı: 0")
        self.squat_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.stats_layout.addWidget(self.squat_label)
        
        # Şınav sayacı
        self.sinav_label = QLabel("Şınav Sayısı: 0")
        self.sinav_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.stats_layout.addWidget(self.sinav_label)
        
        # Kol kaldırma sayacı
        self.kol_label = QLabel("Kol Kaldırma Sayısı: 0")
        self.kol_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.stats_layout.addWidget(self.kol_label)
        
        # Aktif mod bilgisi
        self.aktif_mod_label = QLabel("Aktif Mod: Seçilmedi")
        self.aktif_mod_label.setStyleSheet("font-size: 14px; font-weight: bold; color: red;")
        self.stats_layout.addWidget(self.aktif_mod_label)
        
        self.main_layout.addWidget(self.stats_frame)
        
        # Açıklamalar için alan
        self.explanation_label = QLabel("Hareket analizi durumu burada gösterilecek")
        self.explanation_label.setStyleSheet("font-size: 14px; font-weight: bold; color: blue; padding: 5px;")
        self.explanation_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.explanation_label)
        
        # Kontrol butonları alanı
        self.buttons_layout = QHBoxLayout()
        
        # Webcam başlat butonu
        self.webcam_button = QPushButton("Webcam Başlat")
        self.webcam_button.setStyleSheet("font-size: 14px; padding: 10px; background-color: #4CAF50; color: white;")
        self.webcam_button.clicked.connect(self.start_webcam)
        self.buttons_layout.addWidget(self.webcam_button)
        
        # Video dosyası seç butonu
        self.video_button = QPushButton("Video Seç")
        self.video_button.setStyleSheet("font-size: 14px; padding: 10px; background-color: #2196F3; color: white;")
        self.video_button.clicked.connect(self.select_video)
        self.buttons_layout.addWidget(self.video_button)
        
        # Mod seçimi için combobox
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Squat", "Şınav", "Kol Kaldırma"])
        self.mode_combo.setStyleSheet("font-size: 14px; padding: 10px;")
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        self.buttons_layout.addWidget(self.mode_combo)
        
        # Sayaç sıfırlama butonu
        self.reset_button = QPushButton("Sayacı Sıfırla")
        self.reset_button.setStyleSheet("font-size: 14px; padding: 10px; background-color: #FF9800; color: white;")
        self.reset_button.clicked.connect(self.reset_counter)
        self.buttons_layout.addWidget(self.reset_button)
        
        # Duraklat/Devam et butonu
        self.pause_button = QPushButton("Duraklat/Devam Et")
        self.pause_button.setStyleSheet("font-size: 14px; padding: 10px; background-color: #9C27B0; color: white;")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.buttons_layout.addWidget(self.pause_button)
        
        # Durdur butonu
        self.stop_button = QPushButton("Durdur")
        self.stop_button.setStyleSheet("font-size: 14px; padding: 10px; background-color: #F44336; color: white;")
        self.stop_button.clicked.connect(self.stop_video)
        self.buttons_layout.addWidget(self.stop_button)
        
        self.main_layout.addLayout(self.buttons_layout)
        
        # Başlangıçta bazı butonları devre dışı bırak
        self.reset_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
    def start_webcam(self):
        # Aktif thread varsa durdur
        if self.thread is not None:
            self.stop_video()
            
        # Yeni webcam thread'i başlat
        self.thread = VideoThread(mode="webcam")
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.finished_signal.connect(self.handle_finished)
        self.thread.start()
        
        # Buton durumlarını güncelle
        self.webcam_button.setEnabled(False)
        self.video_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        
        # Durum güncelle
        self.explanation_label.setText("Webcam başlatıldı. Hareket analizi yapılıyor...")
        
    def select_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Video Dosyası Seç", "", "Video Dosyaları (*.mp4 *.avi *.mov *.mkv)")
        
        if file_path:
            # Önce kullanıcıdan egzersiz türünü seçmesini iste
            QMessageBox.information(self, "Egzersiz Türünü Seçin", 
                                "Video yüklendikten sonra, lütfen aşağıdaki mod seçiciden analiz etmek istediğiniz egzersiz türünü seçin.")
            
            # Aktif thread varsa durdur
            if self.thread is not None:
                self.stop_video()
                
            # Yeni video thread'i başlat
            self.thread = VideoThread(mode="video", video_path=file_path)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.finished_signal.connect(self.handle_finished)
            self.thread.start()
            
            # Buton durumlarını güncelle
            self.webcam_button.setEnabled(False)
            self.video_button.setEnabled(False)
            self.reset_button.setEnabled(True)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            
            # Mod combobox'ı vurgula
            self.mode_combo.setStyleSheet("font-size: 14px; padding: 10px; background-color: #FFFF00;")
            
            # Durum güncelle
            video_name = os.path.basename(file_path)
            self.explanation_label.setText(f"Video dosyası başlatıldı: {video_name}. Lütfen egzersiz modunu seçin!")
    
    def change_mode(self, index):
        if self.thread is not None and self.thread.isRunning():
            modes = ["squat", "sinav", "kol_kaldirma"]
            selected_mode = modes[index]
            self.thread.change_mode(selected_mode)
            self.aktif_mod_label.setText(f"Aktif Mod: {selected_mode.capitalize()}")
            self.explanation_label.setText(f"Mod değiştirildi: {selected_mode.capitalize()}")
    
    def reset_counter(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.reset_counter()
            self.explanation_label.setText("Sayaç sıfırlandı.")
    
    def toggle_pause(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.toggle_pause()
            if self.thread.paused:
                self.explanation_label.setText("Video duraklatıldı. Devam etmek için tekrar basın.")
            else:
                self.explanation_label.setText("Video devam ediyor.")
    
    def stop_video(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
            
            # Buton durumlarını güncelle
            self.webcam_button.setEnabled(True)
            self.video_button.setEnabled(True)
            self.reset_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            
            # Durum güncelle
            self.explanation_label.setText("Video durduruldu.")
            
            # Varsayılan görüntüyü göster
            self.display_default_image()
    
    def update_image(self, cv_img, stats):
        """OpenCV görüntüsünü Qt için dönüştür ve göster"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # İstatistikleri güncelle
        self.squat_label.setText(f"Squat Sayısı: {stats['squat_sayisi']}")
        self.sinav_label.setText(f"Şınav Sayısı: {stats['sinav_sayisi']}")
        self.kol_label.setText(f"Kol Kaldırma Sayısı: {stats['kol_kaldirma_sayisi']}")
        self.aktif_mod_label.setText(f"Aktif Mod: {stats['aktif_mod'].capitalize()}")
        
        # Açıklamaları güncelle
        if stats['aciklamalar']:
            self.explanation_label.setText(stats['aciklamalar'])
    
    def handle_finished(self, stats):
        """Video veya webcam bittiğinde çalışır"""
        if "error" in stats:
            QMessageBox.critical(self, "Hata", stats["error"])
        else:
            QMessageBox.information(self, "Tamamlandı", 
                                  f"Analiz tamamlandı.\nSquat: {stats['squat_sayisi']}\n"
                                  f"Şınav: {stats['sinav_sayisi']}\n"
                                  f"Kol Kaldırma: {stats['kol_kaldirma_sayisi']}")
        
        # Buton durumlarını güncelle
        self.webcam_button.setEnabled(True)
        self.video_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        self.thread = None
        self.display_default_image()
        
    def display_default_image(self):
        """Varsayılan bir görüntü göster"""
        self.video_label.clear()
        self.video_label.setText("Webcam veya Video dosyası başlatmak için butonlara tıklayın")
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 16px;")
    
    def closeEvent(self, event):
        """Uygulama kapatılırken çalışan thread'leri durdur"""
        if self.thread is not None:
            self.thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = SporHareketAnaliziApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 