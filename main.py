from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
from PIL import Image
import io
import torch # Render'ın PyTorch'u kurması için gerekli

# Uygulamamızı oluşturuyoruz
app = FastAPI(title="Türk Yemek Tanıma API")

# ==============================================================================
# MODELİN ADRESİ: HUGGING FACE
# ==============================================================================
# Buraya kendi Hugging Face kullanıcı adınızı ve model adınızı yazmalısınız.
# Bu bilgiyi Adım 1'de Hugging Face'e modeli yüklerken belirlemiştik.
# Örnek: "ahmet/turk-yemek-tanima-v1"
MODEL_ADI = "Enesb06/turk-yemek-tanima-v1" 
# ==============================================================================

# Model Yükleme Bloğu
yemek_tanima_modeli = None
try:
    print(f"'{MODEL_ADI}' modeli Hugging Face'den yükleniyor...")
    # Pipeline'a doğrudan modelin adını verdiğimizde, internetten indirip kuracaktır.
    yemek_tanima_modeli = pipeline("image-classification", model=MODEL_ADI)
    print("Model başarıyla yüklendi!")
except Exception as e:
    # Eğer model yüklenemezse, sunucu başlarken loglarda hata basar.
    print(f"!!!!!!!!!!!!!! MODEL YÜKLENEMEDİ !!!!!!!!!!!!!!")
    print(f"HATA: {e}")
    print("Lütfen MODEL_ADI değişkenini doğru yazdığınızdan ve modelinizin Hugging Face'de 'Public' olduğundan emin olun.")
    # Uygulama çökmeyecek ama model kullanılamayacak.

@app.get("/", summary="API Durum Kontrolü")
def read_root():
    """API'nin çalışıp çalışmadığını kontrol eden basit bir endpoint."""
    model_status = "Hazır ve çalışıyor" if yemek_tanima_modeli else "Model yüklenemedi, logları kontrol edin!"
    return {"mesaj": "Türk Yemek Tanıma API'sine hoş geldiniz!", "model_durumu": model_status}

@app.post("/tahmin", summary="Yemek Fotoğrafından Tahmin Yap")
async def tahmin_et(file: UploadFile = File(..., description="Yemek fotoğrafını yükleyin (jpeg, png vb.)")):
    """
    Kullanıcıdan bir resim dosyası alır, AI modeli ile analiz eder
    ve en olası yemek tahminini döndürür.
    """
    if not yemek_tanima_modeli:
        raise HTTPException(status_code=503, detail="Model servisi şu anda aktif değil. Lütfen sunucu loglarını kontrol edin.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir resim dosyası yükleyin.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        print("Tahmin yapılıyor...")
        tahminler = yemek_tanima_modeli(image)
        print(f"Tahmin sonuçları: {tahminler}")
        
        en_iyi_tahmin = tahminler[0]
        yemek_adi = en_iyi_tahmin['label'].replace('_', ' ').title() # 'iskender_kebap' -> 'İskender Kebap'

        return {
            "yemek_adi": yemek_adi,
            "skor": round(en_iyi_tahmin['score'], 4) # Skoru virgülden sonra 4 basamağa yuvarla
        }
    except Exception as e:
        print(f"Tahmin sırasında bir hata oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"Resim işlenirken beklenmedik bir hata oluştu.")