#!/usr/bin/env python3
"""
Скрипт подготовки чистого студийного датасета Светланы с разметкой плюсиками (+) для дообучения в Google Colab.
Вырезает фрагменты из проверенных глав книги, приводит аудио к 24 кГц моно WAV,
расставляет ударения знаком '+' с учетом орфоэпических омографов книги
и формирует архив svetlana_dataset_plus.zip.
"""

import os
import re
import sys
import glob
import zipfile
import subprocess
from pathlib import Path
from ruaccent import RUAccent

BASE_DIR = Path("/workspace/showtime-video-ci")
PROJECT_DIR = BASE_DIR / "projects" / "the-game-of-life-ru"
AUDIO_DIR = BASE_DIR / "local_player" / "audio"
SRT_DIR = PROJECT_DIR / "subtitles"
OUT_DIR = BASE_DIR / "colab_training" / "dataset"
WAV_DIR = OUT_DIR / "wavs"

WAV_DIR.mkdir(parents=True, exist_ok=True)

# Инициализируем акцентуатор
print("📦 Загрузка модели расстановки ударений ruaccent...", flush=True)
accentizer = RUAccent()
accentizer.load(omograph_model_size='tiny', use_dictionary=True)

# Специальные орфоэпические исключения книги (на случай сложных омографов)
SPECIAL_STRESS_MAP = {
    "вдовы": "вдов+ы",
    "вдовы́": "вдов+ы",
    "вдовЫ": "вдов+ы",
    "первообраз": "перво+образ",
    "первоо́браз": "перво+образ",
    "первоОбраз": "перво+образ",
    "первообраза": "перво+образа",
    "первоо́браза": "перво+образа",
    "первоОбраза": "перво+образа",
    "первообразов": "перво+образов",
    "первоо́бразов": "перво+образов",
    "первоОбразов": "перво+образов",
    "сторицей": "стор+ицей",
    "сторИцей": "стор+ицей",
    "обрек": "обр+ёк",
    "обрёк": "обр+ёк",
    "обрётшей": "обр+ётшей",
    "верхом": "в+ерхом",
    "вЕрхом": "в+ерхом",
    "опекунам": "опекун+ам",
    "опекунАм": "опекун+ам",
    "бряцающий": "бряц+ающий",
    "бряцАющий": "бряц+ающий",
    "Господа": "Г+оспода",
    "Го́спода": "Г+оспода",
    "сквозь века": "сквозь век+а",
    "сквозь векА": "сквозь век+а",
    "на краю": "на кра+ю",
    "на краЮ": "на кра+ю",
    "в цепи": "в цеп+и",
    "в цепИ": "в цеп+и",
    "со стороны": "со сторон+ы",
    "со сторонЫ": "со сторон+ы",
    "ростовщику": "ростовщик+у",
    "ростовщикУ": "ростовщик+у",
    "тяжелобольном": "тяжелобольн+ом",
    "тяжелобольнОм": "тяжелобольн+ом",
}

def parse_time(ts_str):
    # '00:03:16,833' -> float seconds
    h, m, s = ts_str.strip().replace(',', '.').split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def parse_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    blocks = re.split(r'\n\s*\n', content.strip())
    cues = []
    for b in blocks:
        lines = b.strip().split('\n')
        if len(lines) >= 3 and '-->' in lines[1]:
            time_parts = lines[1].split('-->')
            start = parse_time(time_parts[0])
            end = parse_time(time_parts[1])
            text = " ".join(lines[2:]).strip()
            cues.append((start, end, text))
    return cues

def accent_text_plus(text):
    # Убираем объединяющие диакритики ударения юникода
    clean_text = text.replace('\u0301', '')
    # Применяем базовую модель ruaccent к тексту
    processed = accentizer.process_all(clean_text)
    
    # ПРИНУДИТЕЛЬНО применяем орфоэпические омографы книги (поверх ruaccent)
    for word, target in SPECIAL_STRESS_MAP.items():
        word_clean = word.replace('+', '').replace('\u0301', '')
        # Регулярка находит слово, даже если ruaccent уже расставил плюсики в других местах (например 'вд+овы')
        chars = [re.escape(c) + r'\+?' for c in word_clean]
        pattern = r'\b' + "".join(chars) + r'\b'
        processed = re.sub(pattern, target, processed, flags=re.IGNORECASE)
        
    return processed

def main():
    print("🚀 Старт формирования датасета Светланы с разметкой плюсиками...", flush=True)
    
    metadata_ljspeech = []
    metadata_sovits = []
    
    clip_count = 0
    total_duration = 0.0
    
    # Обрабатываем все доступные главы (от 01 до 10)
    for ch_num in range(1, 11):
        srt_file = SRT_DIR / f"chapter_{ch_num:02d}.srt"
        audio_file = AUDIO_DIR / f"chapter_{ch_num:02d}.mp3"
        
        if not srt_file.exists() or not audio_file.exists():
            continue
            
        print(f"📖 Обработка Главы {ch_num:02d}...", flush=True)
        cues = parse_srt(srt_file)
        
        for start, end, raw_text in cues:
            duration = end - start
            # Берем фразы от 1.8 до 10 секунд с чистым текстом (оптимально для TTS)
            if duration < 1.8 or duration > 10.0:
                continue
            if len(raw_text) < 15 or len(raw_text) > 160:
                continue
            # Исключаем строки, содержащие только цифры или заголовки
            if re.match(r'^Глава\s+\w+', raw_text):
                continue
                
            clip_id = f"svetlana_{clip_count:05d}"
            out_wav = WAV_DIR / f"{clip_id}.wav"
            
            # Нарезаем аудио и нормализуем к 24 кГц моно PCM 16-bit с микрофейдами
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-i", str(audio_file),
                "-af", f"afade=t=in:ss=0:d=0.03,afade=t=out:st={duration-0.04:.3f}:d=0.04",
                "-ar", "24000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(out_wav)
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size < 1000:
                continue
                
            accented_text = accent_text_plus(raw_text)
            
            # Формат LJSpeech: id|raw|accented
            metadata_ljspeech.append(f"{clip_id}|{raw_text}|{accented_text}")
            # Формат GPT-SoVITS: wavs/clip.wav|speaker|RU|accented
            metadata_sovits.append(f"wavs/{clip_id}.wav|svetlana|RU|{accented_text}")
            
            clip_count += 1
            total_duration += duration
            
            # Для оптимального Colab обучения достаточно 800-1200 идеальных фраз (~1.5 часа)
            if clip_count >= 1000:
                break
                
        if clip_count >= 1000:
            break

    # Сохраняем файлы метаданных
    with open(OUT_DIR / "metadata.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_ljspeech) + "\n")
        
    with open(OUT_DIR / "metadata_sovits.list", "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_sovits) + "\n")
        
    print(f"\n✅ Успешно нарезано {clip_count} клипов ({total_duration/60:.1f} минут чистого аудио Светланы).")
    print(f"📄 Файлы метаданных созданы: metadata.csv и metadata_sovits.list")

    # Упаковываем в компактный zip-архив для Colab
    zip_path = BASE_DIR / "colab_training" / "svetlana_dataset_plus.zip"
    print(f"📦 Создание архива {zip_path.name} для Google Colab...", flush=True)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(OUT_DIR / "metadata.csv", arcname="metadata.csv")
        zipf.write(OUT_DIR / "metadata_sovits.list", arcname="metadata_sovits.list")
        for wav_file in sorted(WAV_DIR.glob("*.wav")):
            zipf.write(wav_file, arcname=f"wavs/{wav_file.name}")
            
    print(f"🎉 АРХИВ ГОТОВ: {zip_path} ({zip_path.stat().st_size / (1024*1024):.1f} MB)")
    print("Готово для загрузки в Google Colab!")

if __name__ == "__main__":
    main()
