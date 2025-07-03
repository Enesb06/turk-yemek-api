from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import io
import torch

app = FastAPI(title="Türk Yemek Tanıma API")

MODEL_ADI = "Enesb06/turk-yemek-tanima-v1" 

yemek_tanima_modeli = None
try:
    print(f"'{MODEL_ADI}' modeli Hugging Face'den yükleniyor...")
    
    processor = AutoImageProcessor.from_pretrained(MODEL_ADI)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ADI)

    yemek_tanima_modeli = pipeline(
        "image-classification", 
        model=model, 
        image_processor=processor,
        device=-1 # Modeli kesinlikle CPU'da çalışmaya zorla
    )
    print("Model başarıyla yüklendi ve CPU'ya atandı!")

except Exception as e:
    print(f"!!!!!!!!!!!!!! MODEL YÜKLENEMEDİ !!!!!!!!!!!!!!")
    print(f"HATA: {e}")
    print("Lütfen MODEL_ADI değişkenini doğru yazdığınızdan ve modelinizin Hugging Face'de 'Public' olduğundan emin olun.")

@app.get("/", summary="API Durum Kontrolü")
def read_root():
    model_status = "Hazır ve çalışıyor" if yemek_tanima_modeli else "Model yüklenemedi, logları kontrol edin!"
    return {"mesaj": "Türk Yemek Tanıma API'sine hoş geldiniz!", "model_durumu": model_status}

@app.post("/tahmin", summary="Yemek Fotoğrafından Tahmin Yap")
async def tahmin_et(file: UploadFile = File(..., description="Yemek fotoğrafını yükleyin (jpeg, png vb.)")):
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
        yemek_adi = en_iyi_tahmin['label'].replace('_', ' ').title()
        return {
            "yemek_adi": yemek_adi,
            "skor": round(en_iyi_tahmin['score'], 4)
        }
    except Exception as e:
        print(f"Tahmin sırasında bir hata oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"Resim işlenirken beklenmedik bir hata oluştu.")
