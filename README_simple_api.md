# Simple MuseTalk API

API simple que llama al script `realtime_inference.py` via subprocess.

## Endpoint único: `/generate`

**POST** `/generate`

### Parámetros (form-data):
- `video`: Archivo de video (MP4, AVI, MOV, etc.)
- `audio`: Archivo de audio (MP3, WAV, M4A)
- `avatar_id`: ID del avatar (opcional, default: "simple_generated")

### Respuesta:
- Archivo de video MP4 generado
- Filename incluye la duración del procesamiento

## Cómo usar:

### 1. Instalar dependencias:
```bash
pip install fastapi uvicorn python-multipart
```

### 2. Ejecutar el API:
```bash
python simple_api.py
```

### 3. Hacer una request:
```bash
curl -X POST "http://localhost:8000/generate" \
  -F "video=@tu_video.mp4" \
  -F "audio=@tu_audio.wav" \
  -F "avatar_id=mi_avatar"
```

## Ventajas:
- ✅ Usa el código original `realtime_inference.py` sin modificaciones
- ✅ Sin problemas de integración compleja
- ✅ Mantiene toda la lógica y optimizaciones del script original
- ✅ Manejo correcto de archivos temporales
- ✅ Simple y confiable

## Desventajas:
- ⚠️ Un poco más lento que integración directa (overhead de subprocess)
- ⚠️ Menos control sobre el proceso interno

## Output:
Los videos generados se guardan en:
```
./results/v15/avatars/{avatar_id}/vid_output/generated.mp4
```

El API automáticamente busca y devuelve el archivo correcto.
