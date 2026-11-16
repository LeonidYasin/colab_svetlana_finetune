import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Monkeypatch transformers.HybridCache для полной совместимости с peft
try:
    import transformers
    if not hasattr(transformers, "HybridCache"):
        transformers.HybridCache = type("HybridCache", (), {})
    try:
        import transformers.cache_utils
        if not hasattr(transformers.cache_utils, "HybridCache"):
            transformers.cache_utils.HybridCache = transformers.HybridCache
    except Exception:
        pass
except Exception:
    pass

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

RU2ARPA = {
    'а': ('AA0', 'AA1'), 'б': ('B',), 'в': ('V',), 'г': ('G',), 'д': ('D',),
    'е': ('EH0', 'EH1'), 'ё': ('Y', 'AO1'), 'ж': ('ZH',), 'з': ('Z',),
    'и': ('IY0', 'IY1'), 'й': ('Y',), 'к': ('K',), 'л': ('L',), 'м': ('M',),
    'н': ('N',), 'о': ('AH0', 'AO1'), 'п': ('P',), 'р': ('R',), 'с': ('S',),
    'т': ('T',), 'у': ('UW0', 'UW1'), 'ф': ('F',), 'х': ('HH',), 'ц': ('T', 'S'),
    'ч': ('CH',), 'ш': ('SH',), 'щ': ('SH',), 'ъ': ('SP',), 'ы': ('IH0', 'IH1'),
    'ь': ('Y',), 'э': ('EH0', 'EH1'), 'ю': ('Y', 'UW0', 'Y', 'UW1'), 'я': ('Y', 'AA0', 'Y', 'AA1'),
}

def ru_text_to_phones(text):
    phones, i, text_lower, stressed = [], 0, text.lower(), False
    while i < len(text_lower):
        ch = text_lower[i]
        if ch == '+':
            stressed = True
            i += 1
            continue
        if ch in RU2ARPA:
            mapping = RU2ARPA[ch]
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
        elif ch in ('.', ',', '!', '?', '…', '-', '。', '，', '！', '？'):
            phones.append('.' if ch == '。' else (',' if ch == '，' else ch))
            stressed = False
        elif ch == ' ':
            phones.append('SP')
            stressed = False
        i += 1
    return phones

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
    for item in ["feature_extractor", "module", "AR", "BigVGAN", "eres2net", "f5_tts", "text"]:
        src = gpt_dir / item
        dst = BASE_DIR / item
        if src.exists() and not dst.exists():
            try:
                shutil.copytree(src, dst)
            except Exception:
                pass
    for item in ["utils.py", "process_ckpt.py", "sv.py"]:
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
    for pkg in [
        "librosa", "scipy", "tensorboard", "pytorch_lightning", "tqdm",
        "einops", "rotary_embedding_torch", "soundfile", "pyyaml", "psutil",
        "numba", "x_transformers", "matplotlib", "torchmetrics", "peft",
        "split_lang", "fast_langdetect", "wordsegment", "resampy",
        "sentencepiece", "g2p_en", "ffmpeg", "jieba", "cn2an", "pypinyin"
    ]:
        try:
            if pkg == "peft":
                from peft import LoraConfig, get_peft_model
            else:
                __import__(pkg)
        except Exception:
            pip_name = "peft==0.12.0" if pkg == "peft" else ("ffmpeg-python" if pkg == "ffmpeg" else pkg.replace("_", "-"))
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name])
    try:
        import pandas
        assert getattr(pandas, "__version__", "").startswith("2.1")
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy<2", "pandas==2.1.4"])

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

    patch_environment()

    try:
        import transformers
        if not hasattr(transformers, "HybridCache"):
            transformers.HybridCache = type("HybridCache", (), {})
        try:
            import transformers.cache_utils
            if not hasattr(transformers.cache_utils, "HybridCache"):
                transformers.cache_utils.HybridCache = transformers.HybridCache
        except Exception:
            pass
    except Exception:
        pass

    # Гарантируем наличие библиотек для инференса речи
    for pkg in ["peft", "split_lang", "fast_langdetect", "wordsegment", "resampy", "sentencepiece", "g2p_en", "ffmpeg", "jieba", "cn2an", "pypinyin"]:
        try:
            if pkg == "ffmpeg":
                import ffmpeg
            elif pkg == "peft":
                from peft import LoraConfig, get_peft_model
            else:
                __import__(pkg)
        except Exception:
            pip_name = "peft==0.12.0" if pkg == "peft" else ("ffmpeg-python" if pkg == "ffmpeg" else pkg.replace("_", "-"))
            print(f"📦 [TTS] Доустановка {pip_name} для озвучки...")
            send_log(f"📦 Установка пакета {pip_name} для синтеза...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name])

    sovits = sorted(Path("SoVITS_weights_v2").glob("svetlana_plus*.pth"), key=os.path.getmtime)
    gpt = sorted(Path("GPT_weights_v2").glob("svetlana_plus*.ckpt"), key=os.path.getmtime)
    if not sovits or not gpt:
        send_log("⚠️ Веса моделей не найдены в SoVITS_weights_v2 или GPT_weights_v2!")
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
        if "ru" not in cfg.languages:
            cfg.languages.extend(["ru", "all_ru"])
        if hasattr(cfg, "v2_languages") and "ru" not in cfg.v2_languages:
            cfg.v2_languages.extend(["ru", "all_ru"])

        send_log(f"🎯 Загрузка моделей: SoVITS={sovits[-1].name}, GPT={gpt[-1].name}")
        tts_engine = TTS(cfg)
        if "ru" not in tts_engine.configs.languages:
            tts_engine.configs.languages.extend(["ru", "all_ru"])

        # Интеграция фонетизатора русской речи с ударениями (+)
        from text import cleaned_text_to_sequence
        orig_segment = tts_engine.text_preprocessor.segment_and_extract_feature_for_text
        orig_pre_seg = tts_engine.text_preprocessor.pre_seg_text

        def custom_segment(text, language, version="v2"):
            if language in ("ru", "all_ru"):
                phones_arpa = ru_text_to_phones(text)
                phones = cleaned_text_to_sequence(phones_arpa, version)
                bert = torch.zeros((1024, len(phones)), dtype=torch.float32).to(tts_engine.text_preprocessor.device)
                return phones, bert, text
            return orig_segment(text, language, version)

        def custom_pre_seg(text, lang, text_split_method):
            if lang in ("ru", "all_ru"):
                raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?…])\s+', text.strip()) if s.strip()]
                if not raw_sentences:
                    raw_sentences = [text.strip()]
                res = []
                for s in raw_sentences:
                    if s and s[-1] not in ('.', '!', '?', '…', ',', ';', ':', '。', '！', '？'):
                        s = s + '.'
                    res.append(s)
                return res
            return orig_pre_seg(text, lang, text_split_method)

        tts_engine.text_preprocessor.segment_and_extract_feature_for_text = custom_segment
        tts_engine.text_preprocessor.pre_seg_text = custom_pre_seg

        print("✅ Движок инференса Светланы успешно загружен!")
        send_log("✅ Движок синтеза речи Светланы готов к работе!")
        return tts_engine
    except Exception as e:
        import traceback
        err_tb = traceback.format_exc()
        print("Ошибка загрузки TTS:\n", err_tb)
        m = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(e))
        if m:
            missing = m.group(1).split(".")[0]
            print(f"📦 [TTS] Авто-установка недостающего модуля: {missing}...")
            send_log(f"📦 Авто-установка недостающего модуля {missing}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", missing])
            try:
                tts_engine = TTS(cfg)
                if "ru" not in tts_engine.configs.languages:
                    tts_engine.configs.languages.extend(["ru", "all_ru"])
                tts_engine.text_preprocessor.segment_and_extract_feature_for_text = custom_segment
                tts_engine.text_preprocessor.pre_seg_text = custom_pre_seg
                print("✅ Движок инференса Светланы успешно загружен после доустановки модуля!")
                send_log("✅ Движок синтеза речи Светланы готов к работе!")
                return tts_engine
            except Exception as e2:
                print("Ошибка при повторной загрузке TTS:", e2)
        send_log(f"⚠️ Ошибка инициализации TTS: {e}\n{err_tb[-300:]}")
        return None

AVAILABLE_PROMPTS = {
    "svetlana_00010": {
        "file": "svetlana_00010.wav",
        "text": "Сквозь век+а м+удрость остаётся неизм+енной.",
        "name": "Сквозь века (3.7 сек, нейтральный)"
    },
    "svetlana_00007": {
        "file": "svetlana_00007.wav",
        "text": "+Этот д+ар вернётся стор+ицей к том+у, кто отд+ал ег+о с люб+овью.",
        "name": "Этот дар (4.8 сек, выразительный)"
    },
    "svetlana_00002": {
        "file": "svetlana_00002.wav",
        "text": "+Именно в н+ём нах+одится иде+альный перво+образ.",
        "name": "Именно в нём (3.7 сек, спокойный)"
    },
    "svetlana_00000": {
        "file": "svetlana_00000.wav",
        "text": "В подсозн+ании ж+енщины запечатл+елся +образ себ+я как вдов+ы.",
        "name": "В подсознании (4.5 сек)"
    }
}

def clean_and_normalize_audio(audio_data, sr=32000, target_peak_db=-1.0):
    if audio_data is None or len(audio_data) == 0:
        return audio_data
    import numpy as np
    if audio_data.dtype == np.int16:
        float_audio = audio_data.astype(np.float32) / 32768.0
    else:
        float_audio = audio_data.astype(np.float32)
        max_val = np.max(np.abs(float_audio))
        if max_val > 1.0:
            float_audio = float_audio / max_val

    # 1. Удаление постоянного смещения (DC offset)
    float_audio = float_audio - np.mean(float_audio)

    # 2. Мягкие микрофейды 5 мс по краям для устранения щелчков
    fade_len = int(sr * 0.005)
    if len(float_audio) > fade_len * 2:
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        float_audio[:fade_len] *= fade_in
        float_audio[-fade_len:] *= fade_out

    # 3. Пиковая нормализация до target_peak_db (-1 dB = ~0.8912)
    peak = np.max(np.abs(float_audio))
    target_scale = 10.0 ** (target_peak_db / 20.0)
    if peak > 1e-5:
        float_audio = float_audio * (target_scale / peak)

    return np.clip(float_audio * 32767.0, -32768.0, 32767.0).astype(np.int16)

def resolve_ref_audio(preferred_id, text_to_speak):
    chosen_id = preferred_id if preferred_id in AVAILABLE_PROMPTS else "svetlana_00010"

    # Защита от коллизии промпта: если текст для озвучки совпадает с промптом или начинается с него,
    # repetition_penalty (1.35) оштрафует правильные токены и приведет к хрипу и искажению!
    clean_target = text_to_speak.replace("+", "").lower()
    prompt_info = AVAILABLE_PROMPTS[chosen_id]
    prompt_clean = prompt_info["text"].replace("+", "").lower()

    first_words = " ".join(prompt_clean.split()[:2])
    if prompt_clean in clean_target or first_words in clean_target:
        for alt_id in ["svetlana_00010", "svetlana_00007", "svetlana_00002"]:
            alt_clean = AVAILABLE_PROMPTS[alt_id]["text"].replace("+", "").lower()
            alt_first = " ".join(alt_clean.split()[:2])
            if alt_clean not in clean_target and alt_first not in clean_target:
                chosen_id = alt_id
                prompt_info = AVAILABLE_PROMPTS[chosen_id]
                break

    ref_filename = prompt_info["file"]
    ref_text = prompt_info["text"]

    for cand in [
        WAVS_DIR / ref_filename,
        DATASET_DIR / ref_filename,
        DATASET_DIR / "wavs" / ref_filename,
        BASE_DIR / ref_filename,
        BASE_DIR / "reference_audio.wav"
    ]:
        if cand.exists():
            return cand, ref_text, prompt_info.get("name", chosen_id)

    wavs = list(WAVS_DIR.glob("*.wav")) or list(DATASET_DIR.glob("*.wav"))
    if wavs:
        return wavs[0], ref_text, wavs[0].name
    return None, ref_text, chosen_id

def handle_tts_task(task):
    global tts_engine
    req_id = task.get("id")
    text_to_speak = task.get("text", "")

    temperature = float(task.get("temperature", 0.6))
    top_p = float(task.get("top_p", 0.6))
    top_k = int(task.get("top_k", 20))
    speed = float(task.get("speed", 1.0))
    ref_id = str(task.get("ref_id", "auto"))

    ref_wav, ref_text, ref_name = resolve_ref_audio(ref_id, text_to_speak)
    if not ref_wav or not ref_wav.exists():
        send_log("⚠️ Ошибка: Образец эталонного голоса не найден на диске!")
        return

    send_log(f"🎙️ Выполняется синтез Светланы [T={temperature:.2f}, TopP={top_p:.2f}, Промпт: {ref_name}]: «{text_to_speak}»")
    tts = get_tts()
    if not tts:
        send_log("⚠️ Не удалось инициализировать синтез речи!")
        return

    try:
        inputs = {
            "text": text_to_speak,
            "text_lang": "ru",
            "ref_audio_path": str(ref_wav),
            "prompt_text": ref_text,
            "prompt_lang": "ru",
            "top_k": top_k,             # 20 (официальное студийное значение)
            "top_p": top_p,             # 0.60 (отсекает низковероятный шум)
            "temperature": temperature, # 0.60 (чистый шелковистый звук без металлического хрипа)
            "repetition_penalty": 1.35, # Предотвращает зацикливания
            "parallel_infer": False,    # ПОСЛЕДОВАТЕЛЬНОЕ VITS-декодирование: устраняет артефакты грубой нарезки
            "split_bucket": False,      # Строгий естественный порядок фраз
            "fragment_interval": 0.25,  # 250 мс естественная пауза между предложениями
            "speed_factor": speed,
        }
        gen = tts.run(inputs)
        chunks = []
        sr = 32000
        for s_rate, chunk in gen:
            sr = s_rate
            chunks.append(chunk)

        if not chunks:
            raise RuntimeError("Движок синтеза не сгенерировал аудио данные")

        import numpy as np
        audio_data = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        audio_data = clean_and_normalize_audio(audio_data, sr=sr, target_peak_db=-1.0)

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
        send_log(f"✅ Чистая реплика Светланы отправлена на сервер (ID: {req_id}, {len(audio_data)/sr:.2f}с, пик: -1.0 dB)")
        try: tmp_out.unlink()
        except: pass
    except Exception as e:
        import traceback
        err_tb = traceback.format_exc()
        print("Ошибка синтеза реплики:\n", err_tb)
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
    s2_dir = "logs/svetlana_plus"
    os.makedirs(f"{s2_dir}/logs_s2_v2", exist_ok=True)
    os.makedirs(f"{s2_dir}/logs_s1_v2", exist_ok=True)
    os.makedirs("SoVITS_weights_v2", exist_ok=True)
    os.makedirs("GPT_weights_v2", exist_ok=True)

    sovits_files = list(Path("SoVITS_weights_v2").glob("svetlana_plus*.pth"))
    if sovits_files:
        send_log(f"✅ Акустическая модель SoVITS уже обучена ({sovits_files[-1].name})! Пропускаем повторное обучение...", progress=70, step=4, step_name="Обучение GPT (15 эпох на RTX 3060)")
    else:
        s2_cfg_path = find_existing_path(["GPT_SoVITS/configs/s2.json", "configs/s2.json"])
        with open(s2_cfg_path, "r", encoding="utf-8") as f:
            s2 = json.load(f)
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
    for pkg in ["matplotlib", "torchmetrics"]:
        try:
            __import__(pkg)
        except ImportError:
            send_log(f"📦 Установка {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])
    try:
        import pandas
        assert getattr(pandas, "__version__", "").startswith("2.1")
    except Exception:
        send_log("📦 Настройка совместимой версии pandas (2.1.4) для модели GPT...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy<2", "pandas==2.1.4"])
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

        # Автоматическая синхронизация worker.py с сервером
        if poll_count % 3 == 0:
            try:
                req_up = urllib.request.Request(f"{SERVER_URL}/worker.py", headers={"Cache-Control": "no-cache"})
                server_code = urllib.request.urlopen(req_up, timeout=3).read().decode("utf-8")
                my_file = Path(__file__).resolve()
                if len(server_code) > 1000 and server_code.strip() != my_file.read_text(encoding="utf-8").strip():
                    print("🔄 [Worker] Обнаружена новая версия worker.py на сервере! Самообновление и перезапуск...")
                    send_log("🔄 Автообновление воркера Windows...")
                    my_file.write_text(server_code, encoding="utf-8")
                    os.execv(sys.executable, [sys.executable, str(my_file)])
            except Exception:
                pass

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
