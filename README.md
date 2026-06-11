# Türkçe Chunking Projesi (Turkish NLP Chunking)

Türkçe cümleler için isim öbeği (NP), eylem öbeği (VP) ve diğer sözdizimsel öbeklerin otomatik olarak tespit edilmesini sağlayan bir makine öğrenmesi projesi. Model olarak **Conditional Random Fields (CRF)** kullanılmıştır.

---

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Kurulum](#kurulum)
- [Dosya Yapısı](#dosya-yapısı)
- [Veri Formatı (CoNLL)](#veri-formatı-conll)
- [Kullanım](#kullanım)
- [Model ve Özellikler](#model-ve-özellikler)
- [Sonuçlar](#sonuçlar)
- [Örnek Çıktı](#örnek-çıktı)

---

## Proje Hakkında

Bu proje, Doğal Dil İşleme (NLP) dersi kapsamında geliştirilmiştir. Temel görev, Türkçe bir cümledeki her kelimenin hangi sözdizimsel öbeğe ait olduğunu tahmin etmektir.

**Desteklenen öbek etiketleri:**

| Etiket   | Açıklama                        |
|----------|---------------------------------|
| B-NP     | İsim öbeği başlangıcı           |
| I-NP     | İsim öbeği devamı               |
| B-VP     | Eylem öbeği başlangıcı          |
| I-VP     | Eylem öbeği devamı              |
| B-ADVP   | Zarf öbeği başlangıcı           |
| I-ADVP   | Zarf öbeği devamı               |
| B-ADJP   | Sıfat öbeği başlangıcı          |
| B-PP     | Edat öbeği başlangıcı           |
| O        | Herhangi bir öbeğe ait değil    |

Etiketleme **BIO (Beginning-Inside-Outside)** şemasını takip eder: `B-` bir öbeğin başlangıcını, `I-` devamını, `O` ise öbek dışı konumları temsil eder.

---

## Kurulum

### Gereksinimler

- Python 3.8+
- pip

### Bağımlılıkları Yükleme

```bash
pip install sklearn-crfsuite scikit-learn numpy matplotlib seaborn
```

---

## Dosya Yapısı

```
proje/
│
├── chunking_project.py   # Ana Python kodu (eğitim, test, grafik)
├── train.conll           # Eğitim verisi (CoNLL formatı)
├── test.conll            # Test verisi (CoNLL formatı)
│
└── outputs/              # Çalıştırıldıktan sonra oluşur
    ├── confusion_matrix.png
    ├── metrics_bar.png
    └── label_distribution.png
```

---

## Veri Formatı (CoNLL)

Veri dosyaları CoNLL formatında düzenlenmiştir. Her satır bir kelimeyi, boş satırlar ise cümle sınırlarını temsil eder.

```
# text = Dün akşam toplantıdan erken çıkan öğrencinin makaleyi okuduğunu fark ettim.
# columns = ID FORM CHUNK-OUTER CHUNK-INNER CLAUSE
1    Dün          B-ADVP    _    O
2    akşam        I-ADVP    _    O
3    toplantıdan  B-NP      B-RELCL    B-RELCL
...
```

Kullanılan sütunlar: `ID`, `FORM` (kelime), `CHUNK-OUTER` (dıştaki öbek etiketi).

---

## Kullanım

Projeyi çalıştırmak için `train.conll` ve `test.conll` dosyalarının `chunking_project.py` ile aynı dizinde bulunması gerekmektedir.

```bash
python chunking_project.py
```

Program çalıştığında sırasıyla şu adımlar gerçekleşir:

1. Eğitim ve test verisi CoNLL dosyalarından yüklenir.
2. Her kelime için özellik vektörleri oluşturulur.
3. CRF modeli eğitim verisiyle eğitilir.
4. Model test verisi üzerinde değerlendirilir; accuracy, precision, recall ve F1-score hesaplanır.
5. Confusion matrix, metrik bar grafiği ve etiket dağılım grafiği `outputs/` klasörüne kaydedilir.
6. Örnek cümleler üzerinde tahmin yapılarak ekrana yazdırılır.

---

## Model ve Özellikler

### Algoritma

**Conditional Random Fields (CRF)** — `sklearn-crfsuite` kütüphanesi, `lbfgs` optimizasyon algoritması.

Hiperparametreler:
- L1 regularization (`c1`): 0.1
- L2 regularization (`c2`): 0.1
- Maksimum iterasyon: 200

### Özellik Mühendisliği

Model her kelime için aşağıdaki özellikleri kullanır:

- Kelimenin kendisi (küçük harf)
- 1–3 karakter ön eki ve 1–4 karakter son eki
- Türkçe morfolojik ekler (isim, eylem, zarf, sıfat, ilgi ekleri)
- Büyük harf / başlık harfi bilgisi
- Noktalama işareti kontrolü
- Kesme işareti varlığı
- Önceki ve sonraki 1–2 kelimenin bilgileri (bağlam penceresi)
- Cümle başı (BOS) / cümle sonu (EOS) işaretçileri

Türkçenin sondan eklemeli yapısına özel morfolojik özellikler modelin başarımını artırmaktadır.

---

## Sonuçlar

Model, test kümesi üzerinde **%62.3 genel doğruluk (accuracy)** elde etmiştir.

### Sınıf Bazlı Metrikler

| Etiket  | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| B-ADVP  | 1.00      | 0.29   | 0.44     |
| B-NP    | 0.60      | 0.75   | 0.67     |
| B-VP    | 0.67      | 0.62   | 0.64     |
| I-NP    | 0.40      | 0.67   | 0.50     |
| O       | 1.00      | 1.00   | 1.00     |

> Seyrek etiketler (B-ADJP, B-PP, I-ADVP, I-VP), test kümesinin küçüklüğü nedeniyle değerlendirilememiştir.

### Grafikler

Çalıştırma sonucunda `outputs/` dizininde üç grafik oluşturulur:

- **confusion_matrix.png** — Gerçek ve tahmin edilen etiketlerin karışıklık matrisi
- **metrics_bar.png** — Her sınıf için Precision / Recall / F1-Score bar grafiği
- **label_distribution.png** — Gerçek ve tahmin edilen etiket frekanslarının karşılaştırması

---

## Örnek Çıktı

```
Cümle: Güzel bir kitap okudum
──────────────────────────────────────────────────
ID   Kelime               Tahmin
──────────────────────────────────────────────────
1    Güzel                B-NP
2    bir                  I-NP
3    kitap                I-NP
4    okudum               B-VP
──────────────────────────────────────────────────
```
