import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONIOENCODING"] = "utf-8"

import re
import sys
import json
import time
import glob
import shutil
import zipfile
import subprocess
import urllib.request
from pathlib import Path

SERVER_URL = "http://{{SERVER_IP}}:{{PORT}}"
BASE_DIR = Path.cwd().resolve()
DATASET_DIR = BASE_DIR / "dataset_svetlana"
WAVS_DIR = DATASET_DIR / "wavs"
LOGS_DIR = BASE_DIR / "logs" / "svetlana_plus"

tts_engine = None

def patch_environment():
    # 1. Патч my_utils.py для надежной загрузки звука через soundfile + librosa без зависимости от ffmpeg.exe
    utils_path = BASE_DIR / "tools" / "my_utils.py"
    if utils_path.exists():
        try:
            with open(utils_path, "r", encoding="utf-8") as f:
                code = f.read()
            changed = False
            if "import ffmpeg" in code and "except ImportError" not in code:
                code = code.replace(
                    "import ffmpeg\nimport gradio as gr\nimport numpy as np\nimport pandas as pd",
                    "import numpy as np\ntry:\n    import ffmpeg\nexcept ImportError:\n    ffmpeg = None\ntry:\n    import gradio as gr\nexcept ImportError:\n    class DummyGR:\n        @staticmethod\n        def Warning(msg): print(f'[WARN] {msg}')\n    gr = DummyGR()\ntry:\n    import pandas as pd\nexcept ImportError:\n    pd = None"
                )
                changed = True
            if "import soundfile as sf" not in code:
                old_load = "def load_audio(file, sr):"
                new_load = """def load_audio(file, sr):
    file = clean_path(file)
    if os.path.exists(file) is False:
        raise RuntimeError(f"File not found: {file}")
    try:
        import soundfile as sf
        import librosa
        audio, orig_sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        return audio.astype(np.float32)
    except Exception:
        pass"""
                code = code.replace(old_load, new_load)
                changed = True
            if changed:
                with open(utils_path, "w", encoding="utf-8") as f:
                    f.write(code)
                print("✅ [Worker] Применен патч tools/my_utils.py (звук через soundfile + librosa)")
        except Exception as e:
            print(f"Предупреждение при патче my_utils.py: {e}")

    # 2. Гарантируем доступность модулей GPT_SoVITS в корне рабочей папки
    gpt_dir = BASE_DIR / "GPT_SoVITS"
    for item in ["feature_extractor", "module", "AR"]:
        src = gpt_dir / item
        dst = BASE_DIR / item
        if src.exists() and not dst.exists():
            try:
                shutil.copytree(src, dst)
            except Exception:
                pass
    for item in ["utils.py", "process_ckpt.py"]:
        src = gpt_dir / item
        dst = BASE_DIR / item
        if src.exists() and not dst.exists():
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass

def find_existing_path(candidates):
    for c in candidates:
        p = Path(c)
        if not p.is_absolute():
            p = BASE_DIR / c
        if p.exists():
            return str(p.resolve())
    return str((BASE_DIR / candidates[0]).resolve())

def check_dependencies():
    patch_environment()
    try:
        import numpy
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2", "-q"])
    try:
        import torch
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2", "-q"])
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"])
    try:
        import transformers
        assert getattr(transformers, "is_torch_available")()
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.43.3", "-q"])
    for pkg in ["librosa", "scipy", "tensorboard", "pytorch_lightning", "tqdm", "einops", "rotary_embedding_torch", "soundfile", "pyyaml", "psutil", "numba", "x_transformers"]:
        try:
            __import__(pkg)
        except ImportError:
            pip_name = pkg.replace("_", "-")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name])

def check_has_weights():
    sovits = list(Path("SoVITS_weights_v2").glob("svetlana_plus*.pth"))
    gpt = list(Path("GPT_weights_v2").glob("svetlana_plus*.ckpt"))
    return bool(sovits and gpt)

def send_ping():
    gpu_name = "NVIDIA CUDA"
    vram_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        else:
            gpu_name = "CPU (CUDA не активен)"
    except Exception as e:
        gpu_name = f"Python (ошибка torch: {str(e)[:120]})"
    has_w = check_has_weights()
    data = json.dumps({"gpu_name": gpu_name, "vram_gb": vram_gb, "has_weights": has_w}).encode("utf-8")
    req = urllib.request.Request(f"{SERVER_URL}/api/worker_ping", data=data, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        print("Ошибка связи с сервером при ping:", e)
    return gpu_name, vram_gb

def send_log(msg, progress=None, step=None, step_name=None, status=None, has_weights=None):
    payload = {"msg": msg}
    if progress is not None: payload["progress"] = progress
    if step is not None: payload["step"] = step
    if step_name is not None: payload["step_name"] = step_name
    if status is not None: payload["status"] = status
    if has_weights is not None: payload["has_weights"] = has_weights
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{SERVER_URL}/api/worker_log", data=data, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=15)
    except Exception:
        pass
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def run_command_logged(cmd, env, log_prefix=""):
    send_log(f"▶️ Запуск: {' '.join(cmd[:3])}...")
    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    last_lines = []
    for line in p.stdout:
        l = line.strip()
        if not l:
            continue
        last_lines.append(l)
        if len(last_lines) > 25:
            last_lines.pop(0)
        if any(k in l.lower() for k in ["epoch", "loss", "step", "filtered", "error", "traceback"]):
            m = re.search(r'epoch[:\s]+(\d+)', l, re.I)
            if m:
                ep = int(m.group(1))
                if "sovits" in log_prefix.lower():
                    prog = 40 + int((min(ep, 20) / 20) * 30)
                else:
                    prog = 70 + int((min(ep, 15) / 15) * 25)
                send_log(f"{log_prefix} {l}", progress=prog)
            else:
                send_log(f"{log_prefix} {l}")
        else:
            print(f"{log_prefix} {l}")
    p.wait()
    if p.returncode != 0:
        err_summary = "\n".join(last_lines[-10:])
        raise RuntimeError(f"Команда завершилась с ошибкой (код {p.returncode}):\n{err_summary}")

def get_tts():
    global tts_engine
    if tts_engine is not None:
        return tts_engine
    sovits = sorted(Path("SoVITS_weights_v2").glob("svetlana_plus*.pth"), key=os.path.getmtime)
    gpt = sorted(Path("GPT_weights_v2").glob("svetlana_plus*.ckpt"), key=os.path.getmtime)
    if not sovits or not gpt:
        return None
    try:
        import torch
        sys.path.append(str(BASE_DIR))
        sys.path.append(str(BASE_DIR / "GPT_SoVITS"))
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        cfg = TTS_Config()
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.is_half = torch.cuda.is_available()
        cfg.version = "v2"
        cfg.t2s_weights_path = str(gpt[-1])
        cfg.vits_weights_path = str(sovits[-1])
        tts_engine = TTS(cfg)
        print("✅ Движок инференса Светланы успешно загружен!")
        return tts_engine
    except Exception as e:
        print("Ошибка загрузки TTS:", e)
        return None

def handle_tts_task(task):
    global tts_engine
    req_id = task.get("id")
    text_to_speak = task.get("text", "")
    send_log(f"🎙️ Выполняется синтез речи: «{text_to_speak}»")
    tts = get_tts()
    if not tts:
        send_log("⚠️ Не найдены обученные веса Светланы!")
        return

    ref_wav = WAVS_DIR / "svetlana_00000.wav"
    if not ref_wav.exists():
        wavs = list(WAVS_DIR.glob("*.wav"))
        ref_wav = wavs[0] if wavs else None
    ref_text = "В подсознании женщины запечатлелся образ себя как вдовы."

    try:
        gen = tts.run({
            "text": text_to_speak,
            "text_lang": "ru",
            "ref_audio_path": str(ref_wav),
            "prompt_text": ref_text,
            "prompt_lang": "ru"
        })
        sr, audio_data = next(gen)
        tmp_out = BASE_DIR / f"test_out_{req_id}.wav"
        try:
            import soundfile as sf
            sf.write(str(tmp_out), audio_data, sr)
        except ImportError:
            from scipy.io import wavfile
            wavfile.write(str(tmp_out), sr, audio_data)

        with open(tmp_out, "rb") as f:
            wav_bytes = f.read()

        req = urllib.request.Request(
            f"{SERVER_URL}/api/worker_upload_tts?id={req_id}",
            data=wav_bytes,
            headers={"Content-Type": "audio/wav"}
        )
        urllib.request.urlopen(req, timeout=15)
        send_log(f"✅ Аудио реплики отправлено на сервер (ID: {req_id})")
        try: tmp_out.unlink()
        except: pass
    except Exception as e:
        send_log(f"⚠️ Ошибка синтеза реплики: {e}")

def run_training():
    send_log("🚀 Запуск дообучения Светланы на RTX 3060...", progress=5, step=1, step_name="Подготовка студийного датасета (102 фразы)", status="running")
    patch_environment()

    # 1. Загрузка и распаковка готового студийного датасета с сервера
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    wavs = list(WAVS_DIR.glob("*.wav"))
    if len(wavs) < 100:
        send_log("📥 Загрузка студийного датасета Светланы с сервера (102 WAV клипа 24 кГц)...")
        zip_path = BASE_DIR / "dataset_svetlana.zip"
        try:
            urllib.request.urlretrieve(f"{SERVER_URL}/dataset_svetlana.zip", str(zip_path))
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(BASE_DIR)
            send_log(f"✅ Датасет распакован ({len(list(WAVS_DIR.glob('*.wav')))} WAV файлов)!")
        except Exception as e:
            send_log(f"⚠️ Ошибка загрузки архива датасета: {e}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "3-bert").mkdir(parents=True, exist_ok=True)

    if (DATASET_DIR / "2-name2text.txt").exists():
        shutil.copy(DATASET_DIR / "2-name2text.txt", LOGS_DIR / "2-name2text.txt")
        send_log("✅ Фонетическая разметка ARPA с ударениями (+) синхронизирована!")

    send_log("✅ Студийный датасет Светланы готов!", progress=25, step=2, step_name="Извлечение признаков (HuBERT 32 кГц + семантика)")

    # 2. Извлечение признаков (HuBERT + семантика)
    env = os.environ.copy()
    base_str = str(BASE_DIR.resolve())
    gpt_str = str((BASE_DIR / "GPT_SoVITS").resolve())
    tools_str = str((BASE_DIR / "tools").resolve())
    env["PYTHONPATH"] = f"{base_str};{gpt_str};{tools_str};{env.get('PYTHONPATH', '')}"

    env["inp_text"] = str(DATASET_DIR / "metadata_sovits.list")
    env["inp_wav_dir"] = str(DATASET_DIR)
    env["exp_name"] = "svetlana_plus"
    env["i_part"] = "0"
    env["all_parts"] = "1"
    env["_CUDA_VISIBLE_DEVICES"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["opt_dir"] = str(LOGS_DIR)
    env["cnhubert_base_dir"] = find_existing_path(["GPT_SoVITS/pretrained_models/chinese-hubert-base", "pretrained_models/chinese-hubert-base"])
    env["bert_pretrained_dir"] = find_existing_path(["GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large", "pretrained_models/chinese-roberta-wwm-ext-large"])
    env["pretrained_s2G"] = find_existing_path(["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", "pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"])
    env["s2config_path"] = find_existing_path(["GPT_SoVITS/configs/s2.json", "configs/s2.json"])

    send_log("1/2 Извлечение HuBERT 32 кГц на видеокарте...")
    run_command_logged([sys.executable, "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], env, "[HuBERT]")

    send_log("2/2 Извлечение семантических признаков...")
    try:
        import x_transformers
    except ImportError:
        send_log("📦 Доустановка модуля x-transformers...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "x-transformers"])
    run_command_logged([sys.executable, "GPT_SoVITS/prepare_datasets/3-get-semantic.py"], env, "[Semantic]")

    tsv_parts = list(LOGS_DIR.glob("6-name2semantic-*.tsv"))
    if tsv_parts:
        lines = []
        for p in sorted(tsv_parts):
            with open(p, "r", encoding="utf-8") as f:
                lines.extend([l.strip() for l in f if l.strip()])
        with open(LOGS_DIR / "6-name2semantic.tsv", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    send_log("✅ Признаки датасета извлечены!", progress=40, step=3, step_name="Обучение SoVITS (20 эпох на RTX 3060)")

    # 3. Обучение SoVITS (20 эпох)
    s2_cfg_path = find_existing_path(["GPT_SoVITS/configs/s2.json", "configs/s2.json"])
    with open(s2_cfg_path, "r", encoding="utf-8") as f:
        s2 = json.load(f)
    s2_dir = "logs/svetlana_plus"
    os.makedirs(f"{s2_dir}/logs_s2_v2", exist_ok=True)
    os.makedirs("SoVITS_weights_v2", exist_ok=True)
    s2["train"]["batch_size"] = 8
    s2["train"]["epochs"] = 20
    s2["train"]["save_every_epoch"] = 5
    s2["train"]["if_save_latest"] = 1
    s2["train"]["if_save_every_weights"] = True
    s2["train"]["text_low_lr_rate"] = 0.4
    s2["train"]["gpu_numbers"] = "0"
    s2["train"]["pretrained_s2G"] = find_existing_path(["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", "pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"])
    s2["train"]["pretrained_s2D"] = find_existing_path(["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth", "pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"])
    s2["model"]["version"] = "v2"
    s2["data"]["exp_dir"] = s2["s2_ckpt_dir"] = s2_dir
    s2["save_weight_dir"] = "SoVITS_weights_v2"
    s2["name"] = "svetlana_plus"
    s2["version"] = "v2"
    os.makedirs("TEMP", exist_ok=True)
    with open("TEMP/tmp_s2.json", "w", encoding="utf-8") as f:
        json.dump(s2, f)

    send_log("🚀 Старт обучения акустической модели SoVITS (20 эпох)...")
    run_command_logged([sys.executable, "GPT_SoVITS/s2_train.py", "--config", "TEMP/tmp_s2.json"], env, "[SoVITS]")

    send_log("✅ Обучение SoVITS успешно завершено!", progress=70, step=4, step_name="Обучение GPT (15 эпох на RTX 3060)")

    # 4. Обучение GPT (15 эпох)
    import yaml
    s1_cfg_path = find_existing_path(["GPT_SoVITS/configs/s1longer-v2.yaml", "configs/s1longer-v2.yaml"])
    with open(s1_cfg_path, "r", encoding="utf-8") as f:
        s1 = yaml.load(f, Loader=yaml.FullLoader)
    os.makedirs("GPT_weights_v2", exist_ok=True)
    s1["train"]["batch_size"] = 8
    s1["train"]["epochs"] = 15
    s1["train"]["save_every_n_epoch"] = 5
    s1["train"]["if_save_latest"] = 1
    s1["train"]["if_save_every_weights"] = True
    s1["pretrained_s1"] = find_existing_path(["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"])
    s1["train"]["half_weights_save_dir"] = "GPT_weights_v2"
    s1["train"]["exp_name"] = "svetlana_plus"
    s1["train_semantic_path"] = f"{s2_dir}/6-name2semantic.tsv"
    s1["train_phoneme_path"] = f"{s2_dir}/2-name2text.txt"
    s1["output_dir"] = f"{s2_dir}/logs_s1_v2"
    with open("TEMP/tmp_s1.yaml", "w", encoding="utf-8") as f:
        yaml.dump(s1, f)

    s1_env = env.copy()
    s1_env["_CUDA_VISIBLE_DEVICES"] = "0"
    s1_env["CUDA_VISIBLE_DEVICES"] = "0"
    s1_env["hz"] = "25hz"
    send_log("🚀 Старт обучения семантической модели GPT (15 эпох)...")
    run_command_logged([sys.executable, "GPT_SoVITS/s1_train.py", "--config_file", "TEMP/tmp_s1.yaml"], s1_env, "[GPT]")

    # 5. Упаковка весов и отправка на сервер
    send_log("📦 Упаковка готовых весов Светланы...", progress=97, step=5, step_name="Упаковка и сохранение весов")
    sovits_files = sorted(Path("SoVITS_weights_v2").glob("svetlana_plus*.pth"), key=os.path.getmtime)
    gpt_files = sorted(Path("GPT_weights_v2").glob("svetlana_plus*.ckpt"), key=os.path.getmtime)

    out_zip = BASE_DIR / "svetlana_custom_weights.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        if sovits_files:
            latest_sovits = sovits_files[-1]
            z.write(latest_sovits, f"SoVITS_weights_v2/{latest_sovits.name}")
            send_log(f"В архив добавлен SoVITS: {latest_sovits.name}")
        if gpt_files:
            latest_gpt = gpt_files[-1]
            z.write(latest_gpt, f"GPT_weights_v2/{latest_gpt.name}")
            send_log(f"В архив добавлен GPT: {latest_gpt.name}")
        sample_wav = WAVS_DIR / "svetlana_00000.wav"
        if sample_wav.exists():
            z.write(sample_wav, "reference_audio.wav")

    try:
        with open(out_zip, "rb") as f:
            data = f.read()
        req = urllib.request.Request(f"{SERVER_URL}/api/worker_upload_weights", data=data, headers={"Content-Type": "application/zip"})
        urllib.request.urlopen(req, timeout=30)
        send_log("✅ Веса успешно переданы на сервер управления!", has_weights=True)
    except Exception as e:
        send_log(f"Веса сохранены локально: {out_zip} ({e})", has_weights=True)

    send_log("🎉 ДООБУЧЕНИЕ СВЕТЛАНЫ НА RTX 3060 ПОЛНОСТЬЮ ЗАВЕРШЕНО!", progress=100, step=5, step_name="Готово!", status="completed", has_weights=True)
    try:
        req = urllib.request.Request(f"{SERVER_URL}/api/worker_finish", data=b"{}", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass

print("=" * 65)
print(f"  🎙️ Воркер Светланы запущен. Подключение к {SERVER_URL}...")
print("=" * 65)
check_dependencies()
gpu_name, vram_gb = send_ping()
print()
print("=" * 65)
print("  🎙️ ВОРКЕР СВЕТЛАНЫ АКТИВЕН И ПОДКЛЮЧЕН К СЕРВЕРУ!")
print("=" * 65)
print(f"  🎮 GPU: {gpu_name} ({vram_gb} GB VRAM)")
print(f"  🌐 Панель управления: {SERVER_URL}")
print()
print("  👉 ДАЛЬНЕЙШИЕ ДЕЙСТВИЯ:")
print(f"  1. В открывшемся браузере перейдите на: {SERVER_URL}")
print("  2. Нажмите зеленую кнопку: «🚀 КНОПКА 2. ЗАПУСТИТЬ ДООБУЧЕНИЕ»")
print()
print("  (Окно консоли не закрывайте — здесь в реальном времени идут логи)")
print("=" * 65)

last_handled_tts = None
poll_count = 0
while True:
    try:
        send_ping()
        poll_count += 1
        if poll_count % 6 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] 🟢 Связь с сервером активна ({SERVER_URL}). Ожидание команды...")

        res = urllib.request.urlopen(f"{SERVER_URL}/api/worker_task", timeout=15)
        data = json.loads(res.read().decode("utf-8"))
        if data.get("task") == "start_training":
            print("\n" + "=" * 65)
            print(f"[{time.strftime('%H:%M:%S')}] 🚀 ПОЛУЧЕНА КОМАНДА: ЗАПУСК ДООБУЧЕНИЯ СВЕТЛАНЫ НА RTX 3060!")
            print("=" * 65)
            try:
                urllib.request.urlopen(urllib.request.Request(f"{SERVER_URL}/api/clear_task", data=b"{}", headers={"Content-Type": "application/json"}))
            except Exception:
                pass
            try:
                run_training()
            except Exception as e:
                send_log(f"❌ Ошибка обучения: {e}", status="error")
                print("=" * 65)
                print(f"❌ Ошибка обучения: {e}")
                print("=" * 65)
        tts_task = data.get("tts_task")
        if tts_task and tts_task.get("status") == "pending":
            if tts_task.get("id") != last_handled_tts:
                last_handled_tts = tts_task.get("id")
                handle_tts_task(tts_task)
    except Exception as e:
        print("Ожидание соединения с сервером:", e)
    time.sleep(2)
