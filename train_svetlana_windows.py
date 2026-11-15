#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматизированный скрипт дообучения студийного голоса Светланы (ru-RU-SvetlanaNeural)
с точным контролем ударений (+) на локальной видеокарте NVIDIA (RTX 3060 / 4060 / 3070 и др.) под Windows.

Запуск:
1. Поместите этот скрипт и train_svetlana.bat в папку GPT-SoVITS
2. Запустите train_svetlana.bat (или python train_svetlana_windows.py)
"""

import os
import sys
import json
import yaml
import glob
import shutil
import asyncio
import subprocess
from pathlib import Path

# Определение интерпретатора Python (встроенный в сборку или системный)
if os.path.exists("runtime/python.exe"):
    PYTHON_EXEC = os.path.abspath("runtime/python.exe")
elif os.path.exists("runtime/bin/python.exe"):
    PYTHON_EXEC = os.path.abspath("runtime/bin/python.exe")
else:
    PYTHON_EXEC = sys.executable

BASE_DIR = Path.cwd()
DATASET_DIR = BASE_DIR / "dataset_svetlana"
WAVS_DIR = DATASET_DIR / "wavs"
LOGS_DIR = BASE_DIR / "logs" / "svetlana_plus"

print("=" * 65)
print("🎙️ ДООБУЧЕНИЕ ГОЛОСА СВЕТЛАНЫ С ТОЧНЫМИ УДАРЕНИЯМИ (+) НА RTX 3060")
print("=" * 65)
print(f"📁 Рабочая папка: {BASE_DIR}")
print(f"🐍 Python: {PYTHON_EXEC}")

# 1. Проверка CUDA и видеокарты
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU активен: {gpu_name} ({vram_gb:.1f} ГБ VRAM)")
    else:
        print("⚠️ Внимание: CUDA не обнаружен! Для обучения на RTX 3060 требуется PyTorch с поддержкой CUDA.")
except ImportError:
    print("⚠️ PyTorch не установлен в текущем окружении.")

# 2. Проверка и установка edge-tts
try:
    import edge_tts
except ImportError:
    print("📦 Установка edge-tts для локальной генерации студийного датасета...")
    subprocess.run([PYTHON_EXEC, "-m", "pip", "install", "-q", "edge-tts"], check=True)
    import edge_tts

# 3. Подготовка студийного датасета Светланы (102 клипа с точными ударениями)
DATASET_DIR.mkdir(parents=True, exist_ok=True)
WAVS_DIR.mkdir(parents=True, exist_ok=True)

# Загружаем размеченные фразы книги
raw_data_path = Path(__file__).parent / "raw_data.json"
if raw_data_path.exists():
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
else:
    raw_data = [
        ["В подсознании женщины запечатлелся образ себя как вдовы.", "В подсозн+ании ж+енщины запечатл+елся +образ себ+я как вдов+ы."],
        ["Там в траурных одеждах сидели две вдовы.", "Там в тр+аурных од+еждах сид+ели две вд+овы."],
        ["Именно в нём находится идеальный первообраз.", "+Именно в н+ём нах+одится иде+альный перво+образ."],
        ["Истинный первообраз всякого блага пребывает в Божественном разуме.", "+Истинный перво+образ вс+якого бл+ага пребыв+ает в Бож+ественном р+азуме."],
        ["Соседи могут счесть это верхом неприличия!", "Сос+еди м+огут счесть +это в+ерхом неприл+ичия!"],
        ["Всадник ехал верхом по холмам.", "Вс+адник +ехал верх+ом по холм+ам."],
        ["Опекунам передали управление состоянием.", "Опекун+ам перед+али управл+ение состо+янием."],
        ["Этот дар вернется сторицей к тому, кто отдал его с любовью.", "+Этот д+ар вернётся стор+ицей к том+у, кто отд+ал ег+о с люб+овью."],
        ["Он обрёк себя на поражение сомнениями.", "Он обр+ёк себ+я на пораж+ение сомн+ениями."],
        ["Благодарность исцелила тяжелобольного человека.", "Благод+арность исцел+ила тяжелобольн+ого челов+ека."]
    ]

existing_wavs = list(WAVS_DIR.glob("*.wav"))
if len(existing_wavs) < len(raw_data):
    print(f"\n🚀 Синтез студийных аудиоклипов Светланы (24 кГц моно)... Всего: {len(raw_data)} шт.")

    async def generate_clips():
        sem = asyncio.Semaphore(4)
        async def task(i, clean_txt):
            async with sem:
                clip_id = f"svetlana_{i:05d}"
                mp3_path = DATASET_DIR / f"{clip_id}.mp3"
                wav_path = WAVS_DIR / f"{clip_id}.wav"
                if wav_path.exists() and wav_path.stat().st_size > 1000:
                    return
                for attempt in range(3):
                    try:
                        comm = edge_tts.Communicate(clean_txt, "ru-RU-SvetlanaNeural")
                        await comm.save(str(mp3_path))
                        if mp3_path.exists() and mp3_path.stat().st_size > 0:
                            break
                    except Exception:
                        await asyncio.sleep(1)
                if mp3_path.exists():
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(mp3_path),
                        "-ar", "24000", "-ac", "1", str(wav_path)
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    try:
                        mp3_path.unlink()
                    except Exception:
                        pass
        await asyncio.gather(*[task(i, p[0]) for i, p in enumerate(raw_data)])

    asyncio.run(generate_clips())
    print(f"✅ Успешно подготовлен студийный датасет ({len(list(WAVS_DIR.glob('*.wav')))} файлов)!")
else:
    print(f"✅ Датасет уже содержит {len(existing_wavs)} готовых аудиоклипов.")

# Копируем WAV-файлы также в корень dataset_svetlana/ для гарантированной совместимости
for wav in WAVS_DIR.glob("*.wav"):
    dest = DATASET_DIR / wav.name
    if not dest.exists():
        shutil.copy(wav, dest)

# Создаем списки разметки
meta_csv = []
meta_sovits = []
for i, (clean_txt, accented_txt) in enumerate(raw_data):
    clip_id = f"svetlana_{i:05d}"
    meta_csv.append(f"{clip_id}|{clean_txt}|{accented_txt}")
    meta_sovits.append(f"wavs/{clip_id}.wav|svetlana|RU|{accented_txt}")

with open(DATASET_DIR / "metadata.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(meta_csv) + "\n")
with open(DATASET_DIR / "metadata_sovits.list", "w", encoding="utf-8") as f:
    f.write("\n".join(meta_sovits) + "\n")

# 4. Извлечение признаков (HuBERT 32 кГц + Семантические токены)
print("\n" + "=" * 65)
print("📊 ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ДАТАСЕТА (HuBERT + SSL + Семантика)")
print("=" * 65)

LOGS_DIR.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["inp_text"] = str(DATASET_DIR / "metadata_sovits.list")
env["inp_wav_dir"] = str(DATASET_DIR)
env["exp_name"] = "svetlana_plus"
env["i_part"] = "0"
env["all_parts"] = "1"
env["_CUDA_VISIBLE_DEVICES"] = "0"
env["opt_dir"] = str(LOGS_DIR)
env["cnhubert_base_dir"] = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
env["bert_pretrained_dir"] = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
env["pretrained_s2G"] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
env["s2config_path"] = "GPT_SoVITS/configs/s2.json"
env["PYTHONPATH"] = f".;GPT_SoVITS;{env.get('PYTHONPATH', '')}"

print("1️⃣ Извлечение HuBERT представлений и нарезка 32 кГц...")
subprocess.run([PYTHON_EXEC, "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], env=env, check=True)

print("2️⃣ Извлечение семантических признаков (Semantic Tokens)...")
subprocess.run([PYTHON_EXEC, "GPT_SoVITS/prepare_datasets/3-get-semantic.py"], env=env, check=True)

# Объединение semantic.tsv
tsv_parts = list(LOGS_DIR.glob("6-name2semantic-*.tsv"))
if tsv_parts:
    lines = []
    for p in sorted(tsv_parts):
        with open(p, "r", encoding="utf-8") as f:
            lines.extend([l.strip() for l in f if l.strip()])
    with open(LOGS_DIR / "6-name2semantic.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# Генерация точной русской фонетической разметки для 2-name2text.txt
ru2arpa = {
    'а': ('AA0', 'AA1'), 'б': ('B',), 'в': ('V',), 'г': ('G',), 'д': ('D',),
    'е': ('EH0', 'EH1'), 'ё': ('Y', 'AO1'), 'ж': ('ZH',), 'з': ('Z',),
    'и': ('IY0', 'IY1'), 'й': ('Y',), 'к': ('K',), 'л': ('L',), 'м': ('M',),
    'н': ('N',), 'о': ('AH0', 'AO1'), 'п': ('P',), 'р': ('R',), 'с': ('S',),
    'т': ('T',), 'у': ('UW0', 'UW1'), 'ф': ('F',), 'х': ('HH',), 'ц': ('T', 'S'),
    'ч': ('CH',), 'ш': ('SH',), 'щ': ('SH',), 'ъ': ('SP',), 'ы': ('IH0', 'IH1'),
    'ь': ('Y',), 'э': ('EH0', 'EH1'), 'ю': ('Y', 'UW0', 'Y', 'UW1'), 'я': ('Y', 'AA0', 'Y', 'AA1'),
}

def text_to_phones(text):
    phones, i, text, stressed = [], 0, text.lower(), False
    while i < len(text):
        ch = text[i]
        if ch == '+':
            stressed = True
            i += 1
            continue
        if ch in ru2arpa:
            mapping = ru2arpa[ch]
            if len(mapping) == 2 and isinstance(mapping[0], str) and isinstance(mapping[1], str):
                phones.append(mapping[1] if stressed else mapping[0])
            elif ch == 'ё':
                phones.extend(['Y', 'AO1'])
            elif ch in ('ю', 'я'):
                phones.append('Y')
                phones.append('UW1' if (ch == 'ю' and stressed) else ('UW0' if ch == 'ю' else ('AA1' if stressed else 'AA0')))
            else:
                phones.extend(mapping)
            stressed = False
        elif ch in ('.', ',', '!', '?', '…', '-'):
            phones.append(ch)
            stressed = False
        elif ch == ' ':
            phones.append('SP')
            stressed = False
        i += 1
    return phones

out_lines = []
for i, (clean_txt, accented_txt) in enumerate(raw_data):
    clip_id = f"svetlana_{i:05d}"
    wav_name = f"{clip_id}.wav"
    phones = text_to_phones(accented_txt)
    phones_str = " ".join(phones)
    word2ph = " ".join(["1"] * len(phones))
    out_lines.append(f"{wav_name}\t{phones_str}\t{word2ph}\t{accented_txt}")

with open(LOGS_DIR / "2-name2text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")

print(f"✅ Фонетическая разметка создана ({len(out_lines)} записей)!")

# 5. Запуск дообучения SoVITS (20 эпох) на RTX 3060
print("\n" + "=" * 65)
print("🚀 [1/2] ДООБУЧЕНИЕ SoVITS (Акустическая модель, 20 эпох)...")
print("Время выполнения на RTX 3060: ~4-5 минут")
print("=" * 65)

with open("GPT_SoVITS/configs/s2.json", "r", encoding="utf-8") as f:
    s2_data = json.load(f)

s2_dir = "logs/svetlana_plus"
os.makedirs(f"{s2_dir}/logs_s2_v2", exist_ok=True)
os.makedirs("SoVITS_weights_v2", exist_ok=True)

s2_data["train"]["batch_size"] = 8
s2_data["train"]["epochs"] = 20
s2_data["train"]["text_low_lr_rate"] = 0.4
s2_data["train"]["pretrained_s2G"] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
s2_data["train"]["pretrained_s2D"] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
s2_data["train"]["if_save_latest"] = True
s2_data["train"]["if_save_every_weights"] = True
s2_data["train"]["save_every_epoch"] = 4
s2_data["train"]["gpu_numbers"] = "0"
s2_data["train"]["grad_ckpt"] = False
s2_data["train"]["lora_rank"] = 32
s2_data["model"]["version"] = "v2"
s2_data["data"]["exp_dir"] = s2_data["s2_ckpt_dir"] = s2_dir
s2_data["save_weight_dir"] = "SoVITS_weights_v2"
s2_data["name"] = "svetlana_plus"
s2_data["version"] = "v2"

os.makedirs("TEMP", exist_ok=True)
with open("TEMP/tmp_s2.json", "w", encoding="utf-8") as f:
    json.dump(s2_data, f, indent=2)

subprocess.run([PYTHON_EXEC, "GPT_SoVITS/s2_train.py", "--config", "TEMP/tmp_s2.json"], check=True)
print("✅ Обучение SoVITS успешно завершено!")

# 6. Запуск дообучения GPT (15 эпох) на RTX 3060
print("\n" + "=" * 65)
print("🚀 [2/2] ДООБУЧЕНИЕ GPT (Семантическая модель, 15 эпох)...")
print("Время выполнения на RTX 3060: ~3-4 минуты")
print("=" * 65)

with open("GPT_SoVITS/configs/s1longer-v2.yaml", "r", encoding="utf-8") as f:
    s1_data = yaml.load(f, Loader=yaml.FullLoader)

os.makedirs(f"{s2_dir}/logs_s1", exist_ok=True)
os.makedirs(f"{s2_dir}/logs_s1_v2", exist_ok=True)
os.makedirs("GPT_weights_v2", exist_ok=True)

s1_data["train"]["batch_size"] = 8
s1_data["train"]["epochs"] = 15
s1_data["pretrained_s1"] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
s1_data["train"]["save_every_n_epoch"] = 5
s1_data["train"]["if_save_every_weights"] = True
s1_data["train"]["if_save_latest"] = True
s1_data["train"]["if_dpo"] = False
s1_data["train"]["half_weights_save_dir"] = "GPT_weights_v2"
s1_data["train"]["exp_name"] = "svetlana_plus"
s1_data["train_semantic_path"] = f"{s2_dir}/6-name2semantic.tsv"
s1_data["train_phoneme_path"] = f"{s2_dir}/2-name2text.txt"
s1_data["output_dir"] = f"{s2_dir}/logs_s1_v2"

with open("TEMP/tmp_s1.yaml", "w", encoding="utf-8") as f:
    yaml.dump(s1_data, f, default_flow_style=False)

s1_env = os.environ.copy()
s1_env["_CUDA_VISIBLE_DEVICES"] = "0"
s1_env["hz"] = "25hz"
subprocess.run([PYTHON_EXEC, "GPT_SoVITS/s1_train.py", "--config_file", "TEMP/tmp_s1.yaml"], env=s1_env, check=True)
print("✅ Обучение GPT успешно завершено!")

# 7. Финализация и сборка архива весов
sovits_weights = sorted(glob.glob("SoVITS_weights*/*svetlana*.pth"))
gpt_weights = sorted(glob.glob("GPT_weights*/*svetlana*.ckpt"))

print("\n" + "=" * 65)
print("🎉 ВСЕ МОДЕЛИ УСПЕШНО ДООБУЧЕНЫ!")
print("=" * 65)
print(f"🎯 Готовые веса SoVITS: {sovits_weights}")
print(f"🎯 Готовые веса GPT: {gpt_weights}")

# Упаковываем веса в архив svetlana_custom_weights.zip
import zipfile
zip_path = BASE_DIR / "svetlana_custom_weights.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sovits_weights:
        zf.write(f, os.path.basename(f))
    for f in gpt_weights:
        zf.write(f, os.path.basename(f))
    ref_wav = WAVS_DIR / "svetlana_00000.wav"
    if ref_wav.exists():
        zf.write(ref_wav, "svetlana_00000.wav")

print(f"\n📦 Архив с весами сохранен: {zip_path}")
print("✨ Готово к использованию в локальном плеере и WebUI!")
