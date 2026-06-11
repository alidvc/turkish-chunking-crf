"""
Turkish NLP Chunking Project (2025-2026)
=========================================
Görev: Türkçe cümleler için isim öbeği (NP), eylem öbeği (VP) vb. öbeklerin tespiti (Chunking)
Model: Conditional Random Fields (CRF)
Format: CoNLL
Değerlendirme: F-measure, Precision, Recall, Accuracy, Confusion Matrix
"""

import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib # type: ignore
matplotlib.use('Agg')
import seaborn as sns # type: ignore
import sklearn_crfsuite # type: ignore
from sklearn.metrics import ( # type: ignore
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. VERİ OKUMA (CoNLL Formatı)
# ─────────────────────────────────────────────

def load_conll(filepath):
    """CoNLL formatındaki dosyayı okur.
    Her cümleyi (tokens, chunk_labels) çifti olarak döndürür.
    """
    sentences = []
    tokens = []
    labels = []

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#') or line == '':
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            word = parts[1]
            chunk = parts[2]  # CHUNK-OUTER sütunu
            tokens.append(word)
            labels.append(chunk)

    if tokens:
        sentences.append((tokens, labels))

    return sentences


# ─────────────────────────────────────────────
# 2. ÖZELLİK ÇIKARIMI (Feature Extraction)
# ─────────────────────────────────────────────

def is_uppercase(word):
    return word[0].isupper() if word else False

def get_suffix(word, n):
    return word[-n:] if len(word) >= n else word

def get_prefix(word, n):
    return word[:n] if len(word) >= n else word

# Türkçe yaygın ekler (morfoljik ipuçları)
TURKISH_NP_SUFFIXES = ['nin', 'nın', 'nun', 'nün', 'yi', 'yı', 'yü', 'yu',
                        'de', 'da', 'te', 'ta', 'den', 'dan', 'ten', 'tan',
                        'lar', 'ler', 'lık', 'lik', 'in', 'ın', 'un', 'ün']
TURKISH_VP_SUFFIXES = ['di', 'dı', 'du', 'dü', 'ti', 'tı', 'tu', 'tü',
                        'yor', 'iyor', 'uyor', 'üyor', 'du', 'duk', 'dük',
                        'mış', 'miş', 'muş', 'müş', 'acak', 'ecek', 'ım', 'im',
                        'um', 'üm', 'tik', 'tık', 'dik', 'dık']
TURKISH_ADV_SUFFIXES = ['ca', 'ce', 'ça', 'çe', 'leyin', 'layın']
TURKISH_ADJ_SUFFIXES = ['lı', 'li', 'lu', 'lü', 'sız', 'siz', 'suz', 'süz', 'sal', 'sel']
TURKISH_REL_SUFFIXES = ['an', 'en', 'yan', 'yen', 'dığı', 'diği', 'duğu', 'düğü',
                         'acak', 'ecek', 'ası', 'esi']

def word_features(sent, i):
    """Bir kelimenin özelliklerini çıkarır."""
    word = sent[i]
    w_lower = word.lower()

    # Temel özellikler
    feats = {
        'word': w_lower,
        'word.lower': w_lower,
        'word.upper': is_uppercase(word),
        'word.len': len(word),

        # Ön ekler
        'prefix1': get_prefix(w_lower, 1),
        'prefix2': get_prefix(w_lower, 2),
        'prefix3': get_prefix(w_lower, 3),

        # Son ekler
        'suffix1': get_suffix(w_lower, 1),
        'suffix2': get_suffix(w_lower, 2),
        'suffix3': get_suffix(w_lower, 3),
        'suffix4': get_suffix(w_lower, 4),

        # Morfolojik ipuçları (Türkçe)
        'is_np_suffix':  any(w_lower.endswith(s) for s in TURKISH_NP_SUFFIXES),
        'is_vp_suffix':  any(w_lower.endswith(s) for s in TURKISH_VP_SUFFIXES),
        'is_adv_suffix': any(w_lower.endswith(s) for s in TURKISH_ADV_SUFFIXES),
        'is_adj_suffix': any(w_lower.endswith(s) for s in TURKISH_ADJ_SUFFIXES),
        'is_rel_suffix': any(w_lower.endswith(s) for s in TURKISH_REL_SUFFIXES),

        # Noktalama
        'is_punct': not word.isalnum() and len(word) == 1,

        # Büyük harf (özel isim ipucu)
        'is_title': word.istitle(),
        'has_apostrophe': "'" in word,
    }

    # Önceki kelime
    if i > 0:
        prev = sent[i-1].lower()
        feats.update({
            'prev_word': prev,
            'prev_suffix2': get_suffix(prev, 2),
            'prev_suffix3': get_suffix(prev, 3),
            'prev_is_upper': is_uppercase(sent[i-1]),
        })
    else:
        feats['BOS'] = True  # Beginning of Sentence

    # Bir önceki önceki kelime
    if i > 1:
        prev2 = sent[i-2].lower()
        feats['prev2_word'] = prev2
        feats['prev2_suffix2'] = get_suffix(prev2, 2)

    # Sonraki kelime
    if i < len(sent) - 1:
        nxt = sent[i+1].lower()
        feats.update({
            'next_word': nxt,
            'next_suffix2': get_suffix(nxt, 2),
            'next_suffix3': get_suffix(nxt, 3),
        })
    else:
        feats['EOS'] = True  # End of Sentence

    # Sonraki sonraki kelime
    if i < len(sent) - 2:
        nxt2 = sent[i+2].lower()
        feats['next2_word'] = nxt2
        feats['next2_suffix2'] = get_suffix(nxt2, 2)

    return feats


def sent_to_features(sent):
    return [word_features(sent, i) for i in range(len(sent))]


# ─────────────────────────────────────────────
# 3. VERİ HAZIRLAMA
# ─────────────────────────────────────────────

def prepare_data(sentences):
    X = [sent_to_features(tokens) for tokens, _ in sentences]
    y = [labels for _, labels in sentences]
    return X, y


# ─────────────────────────────────────────────
# 4. MODEL EĞİTİMİ (CRF)
# ─────────────────────────────────────────────

def train_model(X_train, y_train):
    """CRF modelini eğitir."""
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,   # L1 regularization
        c2=0.1,   # L2 regularization
        max_iterations=200,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    return crf


# ─────────────────────────────────────────────
# 5. DEĞERLENDİRME
# ─────────────────────────────────────────────

def evaluate_model(crf, X_test, y_test):
    """Modeli değerlendirir ve metrikleri hesaplar."""
    y_pred = crf.predict(X_test)

    # Düz liste haline getir
    y_test_flat = [tag for sent in y_test for tag in sent]
    y_pred_flat = [tag for sent in y_pred for tag in sent]

    # Etiketler
    labels = sorted(set(y_test_flat + y_pred_flat))
    labels_no_o = [l for l in labels if l != 'O']  # 'O' hariç

    print("\n" + "="*60)
    print("           DEĞERLENDİRME SONUÇLARI")
    print("="*60)

    # Genel accuracy
    acc = accuracy_score(y_test_flat, y_pred_flat)
    print(f"\nGenel Doğruluk (Accuracy): {acc:.4f} ({acc*100:.2f}%)")

    # Sınıf bazlı rapor
    print("\nSınıf Bazlı Metrikler:")
    print("-"*60)
    report = classification_report(y_test_flat, y_pred_flat, digits=4)
    print(report)

    # Her sınıf için ayrı hesaplama
    results = {}
    for label in labels:
        y_true_bin = [1 if t == label else 0 for t in y_test_flat]
        y_pred_bin = [1 if t == label else 0 for t in y_pred_flat]
        if sum(y_true_bin) == 0:
            continue
        p = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        r = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        results[label] = {'precision': p, 'recall': r, 'f1': f}
        print(f"  [{label}]  Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")

    # Makro ortalama
    macro_f1 = f1_score(y_test_flat, y_pred_flat, average='macro', zero_division=0)
    macro_p  = precision_score(y_test_flat, y_pred_flat, average='macro', zero_division=0)
    macro_r  = recall_score(y_test_flat, y_pred_flat, average='macro', zero_division=0)
    print(f"\nMakro Ortalama → Precision={macro_p:.4f}, Recall={macro_r:.4f}, F1={macro_f1:.4f}")

    return y_test_flat, y_pred_flat, labels, results, acc


# ─────────────────────────────────────────────
# 6. GRAFİKLER
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, labels, out_dir):
    """Karışıklık matrisini çizer."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Tahmin Edilen Etiket', fontsize=12)
    ax.set_ylabel('Gerçek Etiket', fontsize=12)
    ax.set_title('Karışıklık Matrisi (Confusion Matrix)\nTürkçe Chunking - CRF Modeli', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[+] Karışıklık matrisi kaydedildi: {path}")
    return path


def plot_metrics_bar(results, acc, out_dir):
    """Her etiket için Precision/Recall/F1 bar grafiği."""
    labels = list(results.keys())
    precisions = [results[l]['precision'] for l in labels]
    recalls    = [results[l]['recall']    for l in labels]
    f1s        = [results[l]['f1']        for l in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='steelblue', alpha=0.85)
    bars2 = ax.bar(x,         recalls,    width, label='Recall',    color='darkorange', alpha=0.85)
    bars3 = ax.bar(x + width, f1s,        width, label='F1-Score',  color='seagreen', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title(f'Sınıf Bazlı Metrikler — CRF Modeli\nGenel Doğruluk: {acc*100:.1f}%', fontsize=13)
    ax.legend(fontsize=11)
    ax.axhline(y=acc, color='red', linestyle='--', alpha=0.5, label=f'Accuracy={acc:.2f}')

    # Değerleri barların üstüne yaz
    for bar in [*bars1, *bars2, *bars3]:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, 'metrics_bar.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[+] Metrik grafiği kaydedildi: {path}")
    return path


def plot_label_distribution(y_true, y_pred, labels, out_dir):
    """Gerçek ve tahmin etiket dağılımları."""
    true_counts = [y_true.count(l) for l in labels]
    pred_counts = [y_pred.count(l) for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, true_counts, width, label='Gerçek', color='cornflowerblue', alpha=0.8)
    ax.bar(x + width/2, pred_counts, width, label='Tahmin', color='salmon', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Token Sayısı')
    ax.set_title('Gerçek vs. Tahmin Edilen Etiket Dağılımı')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'label_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[+] Etiket dağılımı grafiği kaydedildi: {path}")
    return path


# ─────────────────────────────────────────────
# 7. ÖRNEK TAHMİN — tek cümle
# ─────────────────────────────────────────────

def predict_sentence(crf, sentence):
    """Tek bir cümle için öbekleri tahmin eder."""
    tokens = sentence.split()
    features = sent_to_features(tokens)
    pred = crf.predict([features])[0]
    print("\n" + "─"*50)
    print(f"Cümle: {sentence}")
    print("─"*50)
    print(f"{'ID':<4} {'Kelime':<20} {'Tahmin':<15}")
    print("─"*50)
    for i, (tok, tag) in enumerate(zip(tokens, pred), 1):
        print(f"{i:<4} {tok:<20} {tag:<15}")
    print("─"*50)
    return list(zip(tokens, pred))


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    OUT_DIR = 'outputs'
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  TÜRKÇE CHUNKING PROJESİ — CRF MODELİ")
    print("=" * 60)

    # ── Veri Yükleme ──
    print("\n[1/5] Veri yükleniyor...")
    train_sents = load_conll('train.conll')
    test_sents  = load_conll('test.conll')
    print(f"      Eğitim: {len(train_sents)} cümle")
    print(f"      Test  : {len(test_sents)} cümle")

    all_labels = set()
    for _, labels in train_sents + test_sents:
        all_labels.update(labels)
    print(f"      Etiketler: {sorted(all_labels)}")

    # ── Özellik Çıkarımı ──
    print("\n[2/5] Özellikler çıkarılıyor...")
    X_train, y_train = prepare_data(train_sents)
    X_test,  y_test  = prepare_data(test_sents)
    print(f"      Eğitim token sayısı : {sum(len(s) for s in X_train)}")
    print(f"      Test token sayısı   : {sum(len(s) for s in X_test)}")

    # ── Model Eğitimi ──
    print("\n[3/5] CRF modeli eğitiliyor...")
    crf = train_model(X_train, y_train)
    print("      Model eğitimi tamamlandı.")

    # ── Değerlendirme ──
    print("\n[4/5] Model değerlendiriliyor...")
    y_true_flat, y_pred_flat, all_label_list, per_class_results, acc = \
        evaluate_model(crf, X_test, y_test)

    # ── Grafik Üretimi ──
    print("\n[5/5] Grafikler üretiliyor...")
    cm_path   = plot_confusion_matrix(y_true_flat, y_pred_flat, all_label_list, OUT_DIR)
    bar_path  = plot_metrics_bar(per_class_results, acc, OUT_DIR)
    dist_path = plot_label_distribution(y_true_flat, y_pred_flat, all_label_list, OUT_DIR)

    # ── Örnek Tahminler ──
    print("\n" + "="*60)
    print("  ÖRNEK TAHMİNLER")
    print("="*60)
    test_sentences = [
        "Dün akşam toplantıdan erken çıkan öğrencinin makaleyi okuduğunu fark ettim",
        "Güzel bir kitap okudum",
        "Hızla koşan çocuk parkta oyun oynadı",
        "Sabah kahvaltısını hızlıca yapıp okula koştu",
    ]
    for s in test_sentences:
        predict_sentence(crf, s)

    print("\n" + "="*60)
    print("  PROJE TAMAMLANDI")
    print(f"  Çıktı dizini: {OUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
