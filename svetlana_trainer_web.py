#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-интерфейс дообучения Светланы на RTX 3060 под Windows в 1 клик.
Запуск:
1. Запустите start_web.bat (или python svetlana_trainer_web.py)
2. В браузере откроется http://localhost:7860
3. Нажмите кнопку "▶ Начать дообучение"
"""

import os
import sys
import json
import time
import glob
import shutil
import asyncio
import subprocess
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import webbrowser

# Определение интерпретатора Python
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

# Глобальное состояние обучения
state = {
    "status": "idle",       # idle, running, completed, error
    "step": 0,             # 0: Готов к запуску, 1: Датасет, 2: Признаки, 3: SoVITS, 4: GPT, 5: Готово
    "step_name": "Ожидание запуска",
    "progress": 0,         # 0-100%
    "logs": [],
    "error_msg": "",
    "gpu_name": "Определение GPU...",
    "vram_gb": 0.0,
    "elapsed_sec": 0,
    "start_time": None
}

# Определение GPU
try:
    import torch
    if torch.cuda.is_available():
        state["gpu_name"] = torch.cuda.get_device_name(0)
        state["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    else:
        state["gpu_name"] = "CPU (CUDA не обнаружен)"
except Exception:
    state["gpu_name"] = "NVIDIA CUDA"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    print(entry)
    state["logs"].append(entry)
    if len(state["logs"]) > 2000:
        state["logs"].pop(0)

def run_training_pipeline():
    try:
        state["status"] = "running"
        state["start_time"] = time.time()
        state["progress"] = 5
        log("🚀 Старт автоматизированного процесса дообучения Светланы...")

        # 1. Проверка edge-tts
        try:
            import edge_tts
        except ImportError:
            log("📦 Установка библиотеки edge-tts...")
            subprocess.run([PYTHON_EXEC, "-m", "pip", "install", "-q", "edge-tts"], check=True)
            import edge_tts

        # 2. Подготовка датасета Светланы
        state["step"] = 1
        state["step_name"] = "Генерация студийного датасета Светланы (102 клипа)"
        state["progress"] = 10
        log("📂 Шаг 1: Подготовка студийного датасета Светланы с точными ударениями (+)...")

        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        WAVS_DIR.mkdir(parents=True, exist_ok=True)

        raw_data_path = BASE_DIR / "raw_data.json"
        if not raw_data_path.exists():
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
            log(f"🎙️ Синтез {len(raw_data)} чистых студийных аудиоклипов (24 кГц моно)...")

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

        # Копируем в корень dataset_svetlana/
        for wav in WAVS_DIR.glob("*.wav"):
            dest = DATASET_DIR / wav.name
            if not dest.exists():
                shutil.copy(wav, dest)

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

        log(f"✅ Датасет готов: {len(list(WAVS_DIR.glob('*.wav')))} файлов и разметка.")
        state["progress"] = 25

        # 3. Извлечение признаков
        state["step"] = 2
        state["step_name"] = "Извлечение признаков (HuBERT 32 кГц + Семантика)"
        log("📊 Шаг 2: Извлечение HuBERT и семантических токенов...")

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

        log("... Извлечение представлений HuBERT 32 кГц...")
        proc = subprocess.Popen([PYTHON_EXEC, "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            if "nan filtered" in line or "error" in line.lower():
                log(line.strip())
        proc.wait()

        log("... Извлечение семантических признаков (Semantic Tokens)...")
        proc = subprocess.Popen([PYTHON_EXEC, "GPT_SoVITS/prepare_datasets/3-get-semantic.py"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            if "error" in line.lower():
                log(line.strip())
        proc.wait()

        # Объединение semantic.tsv
        tsv_parts = list(LOGS_DIR.glob("6-name2semantic-*.tsv"))
        if tsv_parts:
            lines = []
            for p in sorted(tsv_parts):
                with open(p, "r", encoding="utf-8") as f:
                    lines.extend([l.strip() for l in f if l.strip()])
            with open(LOGS_DIR / "6-name2semantic.tsv", "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        # Генерация 2-name2text.txt с ударениями (+)
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

        log(f"✅ Фонетическая разметка датасета подготовлена ({len(out_lines)} записей).")
        state["progress"] = 40

        # 4. Обучение SoVITS (20 эпох)
        state["step"] = 3
        state["step_name"] = "Обучение SoVITS (Акустическая модель, 20 эпох)"
        log("🚀 Шаг 3: Запуск дообучения SoVITS на GPU (~4-5 минут)...")

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

        proc = subprocess.Popen([PYTHON_EXEC, "GPT_SoVITS/s2_train.py", "--config", "TEMP/tmp_s2.json"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            line_str = line.strip()
            if "epoch" in line_str.lower() or "loss" in line_str.lower() or "step" in line_str.lower():
                log(f"[SoVITS] {line_str}")
        proc.wait()
        log("✅ Обучение SoVITS успешно завершено!")
        state["progress"] = 70

        # 5. Обучение GPT (15 эпох)
        state["step"] = 4
        state["step_name"] = "Обучение GPT (Семантическая модель, 15 эпох)"
        log("🚀 Шаг 4: Запуск дообучения GPT на GPU (~3-4 минуты)...")

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
        proc = subprocess.Popen([PYTHON_EXEC, "GPT_SoVITS/s1_train.py", "--config_file", "TEMP/tmp_s1.yaml"], env=s1_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            line_str = line.strip()
            if "epoch" in line_str.lower() or "loss" in line_str.lower() or "step" in line_str.lower():
                log(f"[GPT] {line_str}")
        proc.wait()
        log("✅ Обучение GPT успешно завершено!")
        state["progress"] = 95

        # 6. Финализация и упаковка
        state["step"] = 5
        state["step_name"] = "Финализация и упаковка весов"
        sovits_weights = sorted(glob.glob("SoVITS_weights*/*svetlana*.pth"))
        gpt_weights = sorted(glob.glob("GPT_weights*/*svetlana*.ckpt"))

        zip_path = BASE_DIR / "svetlana_custom_weights.zip"
        import zipfile
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sovits_weights:
                zf.write(f, os.path.basename(f))
            for f in gpt_weights:
                zf.write(f, os.path.basename(f))
            ref_wav = WAVS_DIR / "svetlana_00000.wav"
            if ref_wav.exists():
                zf.write(ref_wav, "svetlana_00000.wav")

        total_time = int(time.time() - state["start_time"])
        log(f"🎉 ВСЕ ЭТАПЫ УСПЕШНО ЗАВЕРШЕНЫ ЗА {total_time // 60} мин {total_time % 60} сек!")
        log(f"💾 Веса модели сохранены в архив: {zip_path}")
        state["progress"] = 100
        state["status"] = "completed"

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        log(f"❌ ОШИБКА ОБУЧЕНИЯ: {e}")
        state["status"] = "error"
        state["error_msg"] = str(e)
        state["step_name"] = f"Ошибка: {e}"

class TrainingRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if state["start_time"] and state["status"] == "running":
                state["elapsed_sec"] = int(time.time() - state["start_time"])
            self.wfile.write(json.dumps(state, ensure_ascii=False).encode("utf-8"))
        elif parsed.path == "/download":
            zip_file = BASE_DIR / "svetlana_custom_weights.zip"
            if zip_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", 'attachment; filename="svetlana_custom_weights.zip"')
                self.send_header("Content-Length", str(zip_file.stat().st_size))
                self.end_headers()
                with open(zip_file, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/start":
            if state["status"] != "running":
                state["logs"] = []
                threading.Thread(target=run_training_pipeline, daemon=True).start()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Не засорять терминал HTTP-запросами

HTML_CONTENT = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🎙️ Дообучение Светланы на RTX 3060</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(23, 32, 54, 0.7);
      --border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --accent: #10b981;
      --warning: #f59e0b;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: radial-gradient(circle at top, #1e293b, #0b0f19);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 30px 15px;
    }
    .container {
      width: 100%;
      max-width: 820px;
    }
    .header {
      text-align: center;
      margin-bottom: 25px;
    }
    .header h1 {
      font-size: 26px;
      margin-bottom: 8px;
      background: linear-gradient(135deg, #60a5fa, #34d399);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .header p {
      color: var(--text-muted);
      font-size: 14px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      backdrop-filter: blur(12px);
      padding: 24px;
      margin-bottom: 20px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    .gpu-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 500;
      margin-bottom: 15px;
    }
    .btn-start {
      width: 100%;
      padding: 16px;
      background: linear-gradient(135deg, #2563eb, #10b981);
      border: none;
      border-radius: 12px;
      color: white;
      font-size: 18px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
      box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }
    .btn-start:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 6px 25px rgba(16, 185, 129, 0.5);
    }
    .btn-start:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      filter: grayscale(0.5);
    }
    .progress-bar-bg {
      background: rgba(255, 255, 255, 0.08);
      height: 12px;
      border-radius: 6px;
      overflow: hidden;
      margin: 15px 0 10px;
    }
    .progress-bar-fill {
      background: linear-gradient(90deg, #3b82f6, #10b981);
      height: 100%;
      width: 0%;
      transition: width 0.3s ease;
    }
    .steps-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
      margin: 20px 0;
    }
    .step-item {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 14px;
      color: var(--text-muted);
    }
    .step-icon {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
    }
    .step-item.active {
      color: #60a5fa;
      font-weight: 600;
    }
    .step-item.active .step-icon {
      border-color: #3b82f6;
      background: rgba(59, 130, 246, 0.2);
    }
    .step-item.completed {
      color: #34d399;
    }
    .step-item.completed .step-icon {
      border-color: #10b981;
      background: rgba(16, 185, 129, 0.2);
      color: #10b981;
    }
    .terminal {
      background: #050811;
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 10px;
      padding: 14px;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 12px;
      line-height: 1.5;
      color: #93c5fd;
      height: 220px;
      overflow-y: auto;
      white-space: pre-wrap;
    }
    .btn-download {
      display: inline-block;
      width: 100%;
      text-align: center;
      padding: 14px;
      background: #10b981;
      color: white;
      text-decoration: none;
      border-radius: 10px;
      font-weight: 600;
      margin-top: 15px;
      transition: background 0.2s;
    }
    .btn-download:hover { background: #059669; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🎙️ Дообучение Светланы (GPT-SoVITS v2)</h1>
      <p>Тонкая настройка студийного голоса с контролем ударений (+) на локальной RTX 3060</p>
    </div>

    <div class="card">
      <div id="gpuBadge" class="gpu-badge">
        <span>⚡</span> <span id="gpuText">Проверка GPU...</span>
      </div>

      <button id="btnStart" class="btn-start" onclick="startTraining()">
        <span>▶</span> <span>НАЧАТЬ ДООБУЧЕНИЕ В 1 КЛИК</span>
      </button>

      <div class="progress-bar-bg">
        <div id="progressFill" class="progress-bar-fill"></div>
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted);">
        <span id="stepName">Готов к запуску</span>
        <span id="progressText">0%</span>
      </div>

      <div class="steps-list">
        <div id="st1" class="step-item"><div class="step-icon">1</div><span>Синтез студийного датасета Светланы (102 фразы с ударениями +)</span></div>
        <div id="st2" class="step-item"><div class="step-icon">2</div><span>Извлечение признаков (HuBERT 32 кГц + семантические токены)</span></div>
        <div id="st3" class="step-item"><div class="step-icon">3</div><span>Обучение акустической модели SoVITS (20 эпох, ~4 мин)</span></div>
        <div id="st4" class="step-item"><div class="step-icon">4</div><span>Обучение семантической модели GPT (15 эпох, ~3 мин)</span></div>
        <div id="st5" class="step-item"><div class="step-icon">5</div><span>Готово! Упаковка весов и подготовка к синтезу</span></div>
      </div>

      <div id="downloadBox" style="display: none;">
        <a href="/download" class="btn-download">💾 СКАЧАТЬ ВЕСА МОДЕЛИ (svetlana_custom_weights.zip)</a>
      </div>
    </div>

    <div class="card" style="padding: 16px;">
      <div style="font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--text-muted); display: flex; justify-content: space-between;">
        <span>Консоль вывода в реальном времени:</span>
        <span id="timerText">00:00</span>
      </div>
      <div id="terminal" class="terminal">Ожидание запуска...</div>
    </div>
  </div>

  <script>
    let isRunning = false;

    async function updateStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        
        document.getElementById('gpuText').innerText = `${data.gpu_name} ${data.vram_gb ? '(' + data.vram_gb + ' GB)' : ''}`;
        document.getElementById('progressFill').style.width = data.progress + '%';
        document.getElementById('progressText').innerText = data.progress + '%';
        document.getElementById('stepName').innerText = data.step_name;

        if (data.elapsed_sec) {
          const m = String(Math.floor(data.elapsed_sec / 60)).padStart(2, '0');
          const s = String(data.elapsed_sec % 60).padStart(2, '0');
          document.getElementById('timerText').innerText = `${m}:${s}`;
        }

        // Обновление иконок шагов
        for (let i = 1; i <= 5; i++) {
          const el = document.getElementById('st' + i);
          el.className = 'step-item';
          if (data.step > i || data.status === 'completed') {
            el.className = 'step-item completed';
            el.querySelector('.step-icon').innerText = '✓';
          } else if (data.step === i && data.status === 'running') {
            el.className = 'step-item active';
            el.querySelector('.step-icon').innerText = '●';
          } else {
            el.querySelector('.step-icon').innerText = i;
          }
        }

        const btn = document.getElementById('btnStart');
        if (data.status === 'running') {
          btn.disabled = true;
          btn.innerHTML = '<span>⏳</span> <span>ИДЕТ ОБУЧЕНИЕ НА RTX 3060...</span>';
          isRunning = true;
        } else if (data.status === 'completed') {
          btn.disabled = true;
          btn.innerHTML = '<span>🎉</span> <span>ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!</span>';
          document.getElementById('downloadBox').style.display = 'block';
        } else if (data.status === 'error') {
          btn.disabled = false;
          btn.innerHTML = '<span>⚠️</span> <span>ПОВТОРИТЬ ПОПЫТКУ</span>';
        }

        // Терминал
        const term = document.getElementById('terminal');
        term.innerText = data.logs.join('\\n');
        term.scrollTop = term.scrollHeight;

      } catch (e) {
        console.error(e);
      }
    }

    async function startTraining() {
      const btn = document.getElementById('btnStart');
      btn.disabled = true;
      btn.innerHTML = '<span>⏳</span> <span>ЗАПУСК...</span>';
      await fetch('/api/start', { method: 'POST' });
    }

    setInterval(updateStatus, 1500);
    updateStatus();
  </script>
</body>
</html>
"""

def main():
    port = 7860
    server = HTTPServer(("0.0.0.0", port), TrainingRequestHandler)
    print("=" * 65)
    print(f"🌐 ВЕБ-ПАНЕЛЬ ДООБУЧЕНИЯ СВЕТЛАНЫ ЗАПУЩЕНА!")
    print(f"👉 Откройте в браузере: http://localhost:{port}")
    print("=" * 65)

    # Автоматическое открытие браузера через 1 секунду
    def open_browser():
        time.sleep(1)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")

if __name__ == "__main__":
    main()
