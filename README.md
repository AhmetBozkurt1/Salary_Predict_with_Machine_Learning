<img width="1103" alt="Ekran Resmi 2024-06-14 14 00 19" src="https://github.com/AhmetBozkurt1/Salary_Predict_with_Machine_Learning/assets/120393650/d642d651-e06f-4d8b-b7ea-425faa90c37c">

# SALARY PREDICT WITH MACHINE LEARNING

### İŞ PROBLEMİ
☞ Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol oyuncularının maaş tahminleri için bir makine öğrenmesi modeli geliştiriniz.

### VERİ SETİ HİKAYESİ
☞ Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır. Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır. Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır. 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayınlanan 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.

### DEĞİŞKENLER
- **AtBat:** 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
- **Hits:** 1986-1987 sezonundaki isabet sayısı
- **HmRun:** 1986-1987 sezonundaki en değerli vuruş sayısı
- **Runs:** 1986-1987 sezonunda takımına kazandırdığı sayı
- **RBI:** Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
- **Walks:** Karşı oyuncuya yaptırılan hata sayısı
- **Years:** Oyuncunun major liginde oynama süresi (sene)
- **CAtBat:** Oyuncunun kariyeri boyunca topa vurma sayısı
- **CHits:** Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
- **CHmRun:** Oyucunun kariyeri boyunca yaptığı en değerli sayısı
- **CRuns:** Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
- **CRBI:** Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
- **CWalks:** Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
- **League:** Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
- **Division:** 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
- **PutOuts:** Oyun icinde takım arkadaşınla yardımlaşma
- **Assits:** 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
- **Errors:** 1986-1987 sezonundaki oyuncunun hata sayısı
- **Salary:** Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
- **NewLeague:** 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör



### MODEL OLUŞTURMA
- Veri seti keşfedilir ve özelliklerin analizi yapılır.
- Eksik veriler ve aykırı değerler işlenir.
- Özellik mühendisliği adımlarıyla yeni özellikler türetilir.
- Kategorik değişkenler sayısal formata dönüştürülür.
- Model seçimi yapılır ve hiperparametre optimizasyonu gerçekleştirilir.
- En iyi modelin performansı değerlendirilir.


### Gereksinimler
☞ Bu proje çalıştırılmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- lightgbm
- xgboost
- catboost

### Kurulum
☞ Projeyi yerel makinenizde çalıştırmak için şu adımları izleyebilirsiniz:

- GitHub'dan projeyi klonlayın.
- Projeyi içeren dizine gidin ve terminalde `conda env create -f environment.yaml` komutunu çalıştırarak gerekli bağımlılıkları yükleyin.
- Derleyicinizi `conda` ortamına göre ayarlayın.
- Projeyi bir Python IDE'sinde veya Jupyter Notebook'ta açın.
