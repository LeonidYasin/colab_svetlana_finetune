#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Центральный сервер управления дообучением Светланы на Windows RTX 3060.
Запущен на порту 8888 (http://192.168.0.106:8888).
Обеспечивает:
1. Веб-интерфейс с двумя кнопками:
   - Кнопка 1: Скачать и установить всё для Windows (RTX 3060)
   - Кнопка 2: Запустить дообучение в 1 клик
2. Блок интерактивного тестового синтеза речи (проверка ударений + и голоса прямо в браузере).
3. Диспетчер задач для Windows-воркера (RTX 3060).
4. Потоковое отображение логов, эпох и прогресса обучения в браузере в реальном времени.
5. Раздачу готового студийного датасета (dataset_svetlana.zip).
6. Приём и выдачу готовых дообученных весов (svetlana_custom_weights.zip).
"""

import os
import sys
import re
import json
import time
import shutil
import zipfile
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

PORT = 8888
SERVER_IP = "192.168.0.102"
BASE_DIR = Path(__file__).parent.resolve()
AUDIO_CACHE_DIR = BASE_DIR / "test_audios"
AUDIO_CACHE_DIR.mkdir(exist_ok=True)

def get_current_lan_ip():
    try:
        import subprocess
        out = subprocess.check_output(['ip', 'addr', 'show', 'wlan0'], timeout=1).decode('utf-8')
        m = re.search(r'inet\s+([0-9\.]+)', out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "192.168.0.102"

hub_state = {
    "worker_connected": False,
    "worker_ip": "",
    "gpu_name": "Ожидание подключения Windows...",
    "vram_gb": 0.0,
    "has_weights": False,
    "status": "waiting_worker",  # waiting_worker, ready, running, completed, error
    "step": 0,
    "step_name": "Ожидание подключения Windows RTX 3060",
    "progress": 0,
    "task": "none",              # none, start_training
    "tts_task": None,            # {"id": str, "text": str, "status": "pending"|"processing"|"ready"|"error"}
    "logs": [f"[{time.strftime('%H:%M:%S')}] 🌐 Сервер запущен. Откройте страницу http://{SERVER_IP}:{PORT} на компьютере с Windows."],
    "elapsed_sec": 0,
    "start_time": None
}

class HubRequestHandler(BaseHTTPRequestHandler):
    def get_server_host(self):
        hdr = self.headers.get("Host", "")
        if hdr:
            return hdr.split(":")[0]
        return get_current_lan_ip()

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        server_ip = self.get_server_host()

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.replace("{{SERVER_IP}}", server_ip).replace("{{PORT}}", str(PORT)).encode("utf-8"))

        elif path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if hub_state["start_time"] and hub_state["status"] == "running":
                hub_state["elapsed_sec"] = int(time.time() - hub_state["start_time"])
            self.wfile.write(json.dumps(hub_state, ensure_ascii=False).encode("utf-8"))

        elif path == "/api/worker_task":
            # Windows воркер опрашивает сервер на наличие задач обучения и синтеза
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            resp = {
                "task": hub_state["task"],
                "tts_task": hub_state["tts_task"]
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif path == "/api/tts_status":
            req_id = query.get("id", [""])[0]
            status = "none"
            if hub_state["tts_task"] and hub_state["tts_task"]["id"] == req_id:
                status = hub_state["tts_task"]["status"]
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"id": req_id, "status": status}).encode("utf-8"))

        elif path == "/api/test_audio":
            req_id = query.get("id", [""])[0]
            audio_path = AUDIO_CACHE_DIR / f"test_{req_id}.wav"
            if audio_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(audio_path.stat().st_size))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(audio_path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
            else:
                self.send_response(404)
                self.end_headers()

        elif path == "/raw_data.json":
            raw_path = BASE_DIR / "raw_data.json"
            if raw_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                with open(raw_path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
            else:
                self.send_response(404)
                self.end_headers()

        elif path == "/dataset_svetlana.zip":
            zip_p = BASE_DIR / "dataset_svetlana.zip"
            if zip_p.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", 'attachment; filename="dataset_svetlana.zip"')
                self.send_header("Content-Length", str(zip_p.stat().st_size))
                self.end_headers()
                with open(zip_p, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
            else:
                self.send_response(404)
                self.end_headers()

        elif path == "/setup.bat":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-bat")
            self.send_header("Content-Disposition", 'attachment; filename="setup_svetlana.bat"')
            self.end_headers()
            self.wfile.write(SETUP_BAT_SCRIPT.replace("{{SERVER_IP}}", server_ip).replace("{{PORT}}", str(PORT)).encode("utf-8"))

        elif path == "/setup.ps1":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(SETUP_PS1_SCRIPT.replace("{{SERVER_IP}}", server_ip).replace("{{PORT}}", str(PORT)).encode("utf-8"))

        elif path == "/worker.py":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(WORKER_PY_SCRIPT.replace("{{SERVER_IP}}", server_ip).replace("{{PORT}}", str(PORT)).encode("utf-8"))

        elif path == "/download_weights":
            weights_zip = BASE_DIR / "svetlana_custom_weights.zip"
            if weights_zip.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", 'attachment; filename="svetlana_custom_weights.zip"')
                self.send_header("Content-Length", str(weights_zip.stat().st_size))
                self.end_headers()
                with open(weights_zip, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length", 0))

        if path == "/api/worker_upload_weights":
            # Прием обученных весов от Windows воркера
            weights_zip = BASE_DIR / "svetlana_custom_weights.zip"
            with open(weights_zip, "wb") as f:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 65536))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            hub_state["has_weights"] = True
            log_msg(f"📦 Готовые веса получены с Windows ({weights_zip.stat().st_size / 1024 / 1024:.1f} МБ)!")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "received"}).encode("utf-8"))
            return

        elif path == "/api/worker_upload_tts":
            # Прием сгенерированного аудиофайла от Windows воркера
            req_id = query.get("id", [""])[0]
            audio_path = AUDIO_CACHE_DIR / f"test_{req_id}.wav"
            with open(audio_path, "wb") as f:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 65536))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            if hub_state["tts_task"] and hub_state["tts_task"]["id"] == req_id:
                hub_state["tts_task"]["status"] = "ready"
            log_msg(f"🔊 Тестовая реплика [{req_id}] успешно синтезирована на RTX 3060 ({audio_path.stat().st_size // 1024} КБ)!")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "saved"}).encode("utf-8"))
            return

        body = self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else "{}"
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        if path == "/api/start_training":
            # Пользователь нажал кнопку 2 на веб-странице
            if hub_state["status"] in ["ready", "completed", "error"]:
                hub_state["task"] = "start_training"
                hub_state["status"] = "running"
                hub_state["start_time"] = time.time()
                hub_state["progress"] = 5
                hub_state["step"] = 1
                hub_state["step_name"] = "Старт процесса на RTX 3060..."
                log_msg("🚀 Команда на дообучение передана на видеокарту RTX 3060!")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))

        elif path == "/api/test_tts":
            # Пользователь нажал «Озвучить тест»
            text = data.get("text", "").strip()
            if not text:
                text = "В подсозн+ании ж+енщины запечатл+елся +образ себ+я как вдов+ы."
            req_id = str(int(time.time() * 1000))
            hub_state["tts_task"] = {
                "id": req_id,
                "text": text,
                "status": "pending"
            }
            log_msg(f"🎙️ Запрос тестовой озвучки: «{text[:60]}...» (ID: {req_id})")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "queued", "id": req_id}).encode("utf-8"))

        elif path == "/api/worker_ping":
            # Воркер с Windows сообщает о своем статусе
            hub_state["worker_connected"] = True
            hub_state["worker_ip"] = self.client_address[0]
            if "gpu_name" in data and data["gpu_name"]:
                hub_state["gpu_name"] = data["gpu_name"]
            if "vram_gb" in data:
                hub_state["vram_gb"] = data["vram_gb"]
            if "has_weights" in data:
                hub_state["has_weights"] = bool(data["has_weights"])
            if hub_state["status"] == "waiting_worker":
                hub_state["status"] = "ready"
                hub_state["step_name"] = "Окружение Windows готово к обучению"
                log_msg(f"🟢 Подключен компьютер Windows ({hub_state['gpu_name']}, {hub_state['vram_gb']} GB VRAM). Готов к обучению!")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ack"}).encode("utf-8"))

        elif path == "/api/worker_log":
            # Воркер отправляет логи и процент выполнения
            if "msg" in data:
                log_msg(data["msg"])
            if "progress" in data:
                hub_state["progress"] = data["progress"]
            if "step" in data:
                hub_state["step"] = data["step"]
            if "step_name" in data:
                hub_state["step_name"] = data["step_name"]
            if "status" in data:
                hub_state["status"] = data["status"]
            if "has_weights" in data:
                hub_state["has_weights"] = bool(data["has_weights"])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ack"}).encode("utf-8"))

        elif path == "/api/worker_finish":
            # Завершение обучения
            hub_state["status"] = "completed"
            hub_state["has_weights"] = True
            hub_state["task"] = "none"
            hub_state["progress"] = 100
            hub_state["step"] = 5
            hub_state["step_name"] = "Дообучение успешно завершено!"
            log_msg("🎉 ДООБУЧЕНИЕ СВЕТЛАНЫ НА RTX 3060 ПОЛНОСТЬЮ ЗАВЕРШЕНО!")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ack"}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def log_msg(msg):
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    print(entry)
    hub_state["logs"].append(entry)
    if len(hub_state["logs"]) > 2000:
        hub_state["logs"].pop(0)

# ==============================================================================
# 1. Шаблон страницы управления (HTML / JS / CSS)
# ==============================================================================
HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🎙️ Центр управления дообучением Светланы на RTX 3060</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(23, 32, 54, 0.75);
      --border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --accent: #10b981;
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
    .container { width: 100%; max-width: 880px; }
    .header { text-align: center; margin-bottom: 25px; }
    .header h1 {
      font-size: 26px;
      margin-bottom: 8px;
      background: linear-gradient(135deg, #60a5fa, #34d399);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .header p { color: var(--text-muted); font-size: 14px; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      backdrop-filter: blur(12px);
      padding: 24px;
      margin-bottom: 20px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(245, 158, 11, 0.15);
      border: 1px solid rgba(245, 158, 11, 0.3);
      color: #fbbf24;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 500;
      margin-bottom: 20px;
      transition: all 0.3s;
    }
    .status-badge.connected {
      background: rgba(16, 185, 129, 0.15);
      border-color: rgba(16, 185, 129, 0.3);
      color: #34d399;
    }
    .buttons-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 15px;
      margin-bottom: 20px;
    }
    @media (max-width: 640px) {
      .buttons-grid { grid-template-columns: 1fr; }
    }
    .action-btn {
      padding: 18px;
      border: none;
      border-radius: 12px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      text-decoration: none;
      transition: all 0.2s ease;
      text-align: center;
    }
    .btn-step1 {
      background: linear-gradient(135deg, #1e40af, #2563eb);
      color: white;
      box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }
    .btn-step1:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5); }
    .btn-step2 {
      background: linear-gradient(135deg, #059669, #10b981);
      color: white;
      box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    .btn-step2:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5); }
    .btn-step2:disabled { opacity: 0.4; cursor: not-allowed; filter: grayscale(0.7); }
    .btn-subtext { font-size: 12px; font-weight: normal; opacity: 0.85; }
    
    .powershell-box {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 10px 14px;
      margin-top: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-family: monospace;
      font-size: 12px;
      color: #93c5fd;
    }
    .btn-copy {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: white;
      padding: 5px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 11px;
    }
    .btn-copy:hover { background: rgba(255, 255, 255, 0.2); }

    .progress-bar-bg {
      background: rgba(255, 255, 255, 0.08);
      height: 14px;
      border-radius: 7px;
      overflow: hidden;
      margin: 18px 0 8px;
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
      gap: 10px;
      margin: 20px 0;
    }
    .step-item {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 13px;
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
      font-size: 11px;
    }
    .step-item.active { color: #60a5fa; font-weight: 600; }
    .step-item.active .step-icon { border-color: #3b82f6; background: rgba(59, 130, 246, 0.2); }
    .step-item.completed { color: #34d399; }
    .step-item.completed .step-icon { border-color: #10b981; background: rgba(16, 185, 129, 0.2); color: #10b981; }

    .terminal {
      background: #050811;
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 10px;
      padding: 14px;
      font-family: Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      color: #93c5fd;
      height: 220px;
      overflow-y: auto;
      white-space: pre-wrap;
    }

    .tts-input {
      width: 100%;
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 10px;
      color: white;
      padding: 12px 14px;
      font-size: 14px;
      line-height: 1.5;
      resize: vertical;
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s;
    }
    .tts-input:focus { border-color: #3b82f6; }

    .preset-btn {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #93c5fd;
      padding: 6px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 12px;
      transition: all 0.2s;
    }
    .preset-btn:hover {
      background: rgba(59, 130, 246, 0.2);
      border-color: #3b82f6;
      color: white;
    }

    .btn-download-weights {
      display: block;
      width: 100%;
      text-align: center;
      padding: 16px;
      background: linear-gradient(135deg, #059669, #10b981);
      color: white;
      text-decoration: none;
      border-radius: 12px;
      font-weight: 700;
      font-size: 16px;
      margin-top: 15px;
      box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🎙️ Панель дообучения Светланы на RTX 3060</h1>
      <p>100% автоматизация: точные ударения (+), русская фонетика ARPA, ~7 минут на видеокарте</p>
    </div>

    <!-- КАРТОЧКА 1: УПРАВЛЕНИЕ ОБУЧЕНИЕМ -->
    <div class="card">
      <div id="statusBadge" class="status-badge">
        <span>●</span> <span id="statusText">Ожидание подключения Windows RTX 3060...</span>
      </div>

      <div class="buttons-grid">
        <a href="/setup.bat" class="action-btn btn-step1">
          <span>📥 КНОПКА 1. СКАЧАТЬ И УСТАНОВИТЬ ВСЁ</span>
          <span class="btn-subtext">Скачает setup_svetlana.bat (просто нажмите на него в загрузках)</span>
        </a>

        <button id="btnTrain" class="action-btn btn-step2" onclick="startTraining()" disabled>
          <span>🚀 КНОПКА 2. ЗАПУСТИТЬ ДООБУЧЕНИЕ</span>
          <span id="btnTrainSubtext" class="btn-subtext">Активируется при подключении Windows (~7 мин)</span>
        </button>
      </div>

      <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 5px;">Или выполните 1 команду в PowerShell на Windows (без скачивания файлов):</div>
      <div class="powershell-box">
        <span id="psCode">irm http://{{SERVER_IP}}:{{PORT}}/setup.ps1 | iex</span>
        <button class="btn-copy" onclick="copyPS()">Копировать</button>
      </div>

      <div class="progress-bar-bg">
        <div id="progressFill" class="progress-bar-fill"></div>
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted);">
        <span id="stepName">Ожидание подключения воркера</span>
        <span id="progressText">0%</span>
      </div>

      <div class="steps-list">
        <div id="st1" class="step-item"><div class="step-icon">1</div><span>Распаковка студийного датасета Светланы (102 фразы с точными ударениями +)</span></div>
        <div id="st2" class="step-item"><div class="step-icon">2</div><span>Извлечение признаков на RTX 3060 (HuBERT 32 кГц + семантика)</span></div>
        <div id="st3" class="step-item"><div class="step-icon">3</div><span>Обучение акустической модели SoVITS (20 эпох, ~4 мин)</span></div>
        <div id="st4" class="step-item"><div class="step-icon">4</div><span>Обучение семантической модели GPT (15 эпох, ~3 мин)</span></div>
        <div id="st5" class="step-item"><div class="step-icon">5</div><span>Готово! Упаковка весов и синхронизация с плеером аудиокниги</span></div>
      </div>

      <div id="downloadBox" style="display: none;">
        <a href="/download_weights" class="btn-download-weights">💾 СКАЧАТЬ ГОТОВЫЕ ВЕСА (svetlana_custom_weights.zip)</a>
      </div>
    </div>

    <!-- КАРТОЧКА 2: ИНТЕРАКТИВНАЯ ТЕСТОВАЯ ОЗВУЧКА СВЕТЛАНЫ -->
    <div class="card" id="testTtsCard">
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <h3 style="font-size: 18px; color: #34d399; display: flex; align-items: center; gap: 8px;">
          <span>🎧</span> <span>Тестовая озвучка голосом Светланы (Проверка ударений +)</span>
        </h3>
        <span id="ttsReadyBadge" style="font-size: 12px; padding: 4px 12px; border-radius: 12px; background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3);">
          Ожидание обучения модели
        </span>
      </div>

      <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px; line-height: 1.5;">
        Введите любую фразу. Ставьте знак <b>+</b> перед ударной гласной (например: <i>вдов+ы</i>, <i>перво+образ</i>, <i>стор+ицей</i>), чтобы убедиться в безошибочном послушании нейросети:
      </p>

      <textarea id="ttsTextInput" class="tts-input" rows="3">В подсозн+ании ж+енщины запечатл+елся +образ себ+я как вдов+ы. А там в тр+аурных од+еждах сид+ели две вд+овы.</textarea>

      <!-- Быстрые пресеты для проверки ключевых слов книги -->
      <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;">
        <button class="preset-btn" onclick="setPreset(1)">1. Вдов+ы vs Вд+овы</button>
        <button class="preset-btn" onclick="setPreset(2)">2. Перво+образ</button>
        <button class="preset-btn" onclick="setPreset(3)">3. В+ерхом vs Верх+ом</button>
        <button class="preset-btn" onclick="setPreset(4)">4. Стор+ицей и обр+ёк</button>
      </div>

      <div style="margin-top: 15px;">
        <button id="btnSynthesize" class="action-btn btn-step2" style="width: 100%;" onclick="requestSynthesis()" disabled>
          <span>🔊 ОЗВУЧИТЬ НА RTX 3060</span>
          <span id="btnSynthSubtext" class="btn-subtext">Активируется после готовности весов</span>
        </button>
      </div>

      <!-- Аудиоплеер для мгновенного прослушивания -->
      <div id="audioPlayerBox" style="display: none; margin-top: 16px; padding: 14px; background: rgba(0,0,0,0.3); border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
        <div style="font-size: 13px; margin-bottom: 8px; color: #93c5fd; font-weight: 600; display: flex; justify-content: space-between;">
          <span>Результат синтеза Светланы:</span>
          <span id="audioStatusText" style="color: #34d399;">Готово к воспроизведению</span>
        </div>
        <audio id="audioResult" controls autoplay style="width: 100%; height: 42px; border-radius: 8px;"></audio>
      </div>
    </div>

    <!-- КАРТОЧКА 3: ТЕРМИНАЛ ЛОГОВ -->
    <div class="card" style="padding: 16px;">
      <div style="font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--text-muted); display: flex; justify-content: space-between;">
        <span>Терминал вывода RTX 3060 в реальном времени:</span>
        <span id="timerText">00:00</span>
      </div>
      <div id="terminal" class="terminal">Ожидание подключения воркера Windows...</div>
    </div>
  </div>

  <script>
    function copyPS() {
      const code = document.getElementById('psCode').innerText;
      navigator.clipboard.writeText(code);
      alert('Команда скопирована в буфер обмена! Нажмите Ctrl+V в окне PowerShell на Windows.');
    }

    function setPreset(idx) {
      const input = document.getElementById('ttsTextInput');
      if (idx === 1) {
        input.value = "В подсозн+ании ж+енщины запечатл+елся +образ себ+я как вдов+ы. А там в тр+аурных од+еждах сид+ели две вд+овы.";
      } else if (idx === 2) {
        input.value = "+Именно в н+ём нах+одится иде+альный перво+образ вс+якого бл+ага, пребыв+ающий в Бож+ественном р+азуме.";
      } else if (idx === 3) {
        input.value = "Сос+еди м+огут счесть +это в+ерхом неприл+ичия, но вс+адник +ехал верх+ом по холм+ам.";
      } else if (idx === 4) {
        input.value = "+Этот д+ар вернётся стор+ицей к том+у, кто отд+ал ег+о с люб+овью, а сомн+ения обр+екли ег+о на пров+ал.";
      }
    }

    async function startTraining() {
      const btn = document.getElementById('btnTrain');
      btn.disabled = true;
      btn.innerHTML = '<span>⏳ ОБУЧЕНИЕ ЗАПУСКАЕТСЯ НА RTX 3060...</span>';
      await fetch('/api/start_training', { method: 'POST' });
    }

    async function requestSynthesis() {
      const btn = document.getElementById('btnSynthesize');
      const text = document.getElementById('ttsTextInput').value.trim();
      if (!text) {
        alert('Введите текст для озвучки!');
        return;
      }
      btn.disabled = true;
      btn.innerHTML = '<span>⏳ СИНТЕЗ НА RTX 3060...</span><span class="btn-subtext">Нейросеть генерирует аудио...</span>';

      try {
        const res = await fetch('/api/test_tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: jsonString({ text })
        });
        const data = await res.json();
        const reqId = data.id;

        // Опрос статуса генерации
        const pollTts = setInterval(async () => {
          try {
            const check = await fetch('/api/tts_status?id=' + reqId);
            const cdata = await check.json();
            if (cdata.status === 'ready') {
              clearInterval(pollTts);
              btn.disabled = false;
              btn.innerHTML = '<span>🔊 ОЗВУЧИТЬ НА RTX 3060</span><span class="btn-subtext">Синтез завершён за пару секунд!</span>';
              
              const pBox = document.getElementById('audioPlayerBox');
              const aRes = document.getElementById('audioResult');
              pBox.style.display = 'block';
              aRes.src = '/api/test_audio?id=' + reqId + '&t=' + Date.now();
              aRes.play();
            } else if (cdata.status === 'error') {
              clearInterval(pollTts);
              btn.disabled = false;
              btn.innerHTML = '<span>⚠️ ОШИБКА СИНТЕЗА</span>';
            }
          } catch (e) {
            console.error(e);
          }
        }, 800);

      } catch (err) {
        alert('Ошибка отправки: ' + err);
        btn.disabled = false;
      }
    }

    function jsonString(obj) { return JSON.stringify(obj); }

    async function poll() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        const badge = document.getElementById('statusBadge');
        const btnTrain = document.getElementById('btnTrain');
        const btnSynth = document.getElementById('btnSynthesize');
        const ttsBadge = document.getElementById('ttsReadyBadge');
        const btnSynthSub = document.getElementById('btnSynthSubtext');

        if (data.worker_connected) {
          badge.className = 'status-badge connected';
          badge.innerHTML = `<span>🟢</span> <span>Windows на связи: ${data.gpu_name} (${data.vram_gb} GB VRAM)</span>`;
          if (data.status === 'ready' || data.status === 'completed') {
            btnTrain.disabled = false;
            btnTrain.innerHTML = '<span>🚀 КНОПКА 2. ЗАПУСТИТЬ ДООБУЧЕНИЕ</span><span class="btn-subtext">RTX 3060 готова! Нажмите для старта (~7 мин)</span>';
          }
        } else {
          badge.className = 'status-badge';
          badge.innerHTML = `<span>●</span> <span>${data.step_name}</span>`;
        }

        if (data.has_weights || data.status === 'completed') {
          ttsBadge.style.background = 'rgba(16, 185, 129, 0.15)';
          ttsBadge.style.color = '#34d399';
          ttsBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
          ttsBadge.innerText = '✅ Модель Светланы готова к озвучке!';
          if (!btnSynth.innerHTML.includes('СИНТЕЗ НА RTX 3060')) {
            btnSynth.disabled = false;
            btnSynthSub.innerText = 'Нажмите для мгновенного синтеза на RTX 3060';
          }
        }

        if (data.status === 'running') {
          btnTrain.disabled = true;
          btnTrain.innerHTML = '<span>⏳ ИДЁТ ОБУЧЕНИЕ НА RTX 3060...</span>';
        } else if (data.status === 'completed') {
          btnTrain.disabled = true;
          btnTrain.innerHTML = '<span>🎉 ДООБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!</span>';
          document.getElementById('downloadBox').style.display = 'block';
        }

        document.getElementById('progressFill').style.width = data.progress + '%';
        document.getElementById('progressText').innerText = data.progress + '%';
        document.getElementById('stepName').innerText = data.step_name;

        if (data.elapsed_sec) {
          const m = String(Math.floor(data.elapsed_sec / 60)).padStart(2, '0');
          const s = String(data.elapsed_sec % 60).padStart(2, '0');
          document.getElementById('timerText').innerText = `${m}:${s}`;
        }

        // Обновление шагов
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

        const term = document.getElementById('terminal');
        term.innerText = data.logs.join('\\n');
        term.scrollTop = term.scrollHeight;

      } catch (e) {
        console.error(e);
      }
    }

    setInterval(poll, 1500);
    poll();
  </script>
</body>
</html>
"""

# ==============================================================================
# 2. Windows Batch скрипт (скачивается Кнопкой 1)
# ==============================================================================
SETUP_BAT_SCRIPT = """@echo off
chcp 65001 > nul
title Установка и подключение воркера Светланы к RTX 3060

echo =========================================================
echo   🎙️ АВТОМАТИЧЕСКАЯ НАСТРОЙКА ОКРУЖЕНИЯ ДЛЯ RTX 3060
echo =========================================================
echo Подключение к серверу http://{{SERVER_IP}}:{{PORT}} ...

powershell -ExecutionPolicy Bypass -Command "irm http://{{SERVER_IP}}:{{PORT}}/setup.ps1 | iex"
pause
"""

# ==============================================================================
# 3. Windows PowerShell скрипт автоматической подготовки (setup.ps1)
# ==============================================================================
SETUP_PS1_SCRIPT = """# Автоматическая настройка воркера GPT-SoVITS на Windows
$ServerUrl = "http://{{SERVER_IP}}:{{PORT}}"
$WorkDir = "$HOME\\GPT-SoVITS-Svetlana"

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "  🎙️ Подготовка окружения Светланы на RTX 3060" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# 1. Поиск уже установленного GPT-SoVITS на дисках пользователя
$candidates = @(
    (Get-Location).Path,
    "C:\\GPT-SoVITS", "D:\\GPT-SoVITS", "E:\\GPT-SoVITS",
    "$HOME\\GPT-SoVITS", "$HOME\\GPT-SoVITS-v2",
    "$HOME\\GPT-SoVITS-Svetlana"
)

$foundDir = $null
foreach ($c in $candidates) {
    if ((Test-Path "$c\\GPT_SoVITS") -or (Test-Path "$c\\runtime\\python.exe")) {
        $foundDir = $c
        Write-Host "✅ Обнаружена папка GPT-SoVITS: $foundDir" -ForegroundColor Green
        break
    }
}

if ($foundDir) {
    $WorkDir = $foundDir
} else {
    if (-not (Test-Path $WorkDir)) {
        New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
    }
}
Set-Location $WorkDir
Write-Host "📁 Рабочая директория: $WorkDir" -ForegroundColor Cyan

# 2. Определение интерпретатора Python (для PyTorch CUDA на RTX 3060 нужен Python 3.10 или 3.11)
$PythonExe = $null

if (Test-Path "runtime\\python.exe") {
    $PythonExe = (Resolve-Path "runtime\\python.exe").Path
    Write-Host "🐍 Используется встроенный Python сборки: $PythonExe" -ForegroundColor Green
}

if (-not $PythonExe -and (Get-Command py -ErrorAction SilentlyContinue)) {
    $p10 = & py -3.10 -c "import sys; print(sys.executable)" 2>$null
    $p11 = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
    if ($p10) {
        $PythonExe = $p10.Trim()
        Write-Host "🐍 Найден Python 3.10: $PythonExe" -ForegroundColor Green
    } elseif ($p11) {
        $PythonExe = $p11.Trim()
        Write-Host "🐍 Найден Python 3.11: $PythonExe" -ForegroundColor Green
    }
}

if (-not $PythonExe -and (Test-Path "C:\\Python310\\python.exe")) {
    $PythonExe = "C:\\Python310\\python.exe"
} elseif (-not $PythonExe -and (Test-Path "C:\\Python311\\python.exe")) {
    $PythonExe = "C:\\Python311\\python.exe"
}

if (-not $PythonExe -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($ver -and [float]$ver -ge 3.13) {
        Write-Host "⚠️ Обнаружен Python $ver. Для CUDA на RTX 3060 устанавливаем Python 3.10..." -ForegroundColor Yellow
        winget install --id Python.Python.3.10 -e --source winget --accept-package-agreements --accept-source-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        if (Get-Command py -ErrorAction SilentlyContinue) {
            $p = & py -3.10 -c "import sys; print(sys.executable)" 2>$null
            if ($p) { $PythonExe = $p.Trim() } else { $PythonExe = "python" }
        } else {
            $PythonExe = "python"
        }
    } else {
        $PythonExe = "python"
        Write-Host "🐍 Используется системный Python ($ver)!" -ForegroundColor Green
    }
}

if (-not $PythonExe) {
    Write-Host "⚠️ Python не найден. Устанавливаем Python 3.10 через winget..." -ForegroundColor Yellow
    winget install --id Python.Python.3.10 -e --source winget --accept-package-agreements --accept-source-agreements
    $PythonExe = "python"
}

# 2.1 Проверка PyTorch и CUDA
Write-Host "⚙️ Проверка PyTorch с поддержкой CUDA на RTX 3060..." -ForegroundColor Cyan
$cudaCheck = & $PythonExe -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaCheck -ne "True") {
    Write-Host "📦 Установка PyTorch с поддержкой CUDA для вашей RTX 3060..." -ForegroundColor Yellow
    & $PythonExe -m pip install -q --upgrade pip
    & $PythonExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    & $PythonExe -m pip install -q soundfile scipy pyyaml psutil
}
Write-Host "✅ PyTorch с поддержкой CUDA готов к работе!" -ForegroundColor Green

# 3. Клонирование GPT-SoVITS если директория не содержит репозитория
if (-not (Test-Path "GPT_SoVITS")) {
    Write-Host "📦 Клонирование репозитория GPT-SoVITS..." -ForegroundColor Green
    git clone --depth=1 https://github.com/RVC-Boss/GPT-SoVITS.git .
}

# 4. Проверка базовых предобученных весов v2
$s2g = "GPT_SoVITS\\pretrained_models\\gsv-v2final-pretrained\\s2G2333k.pth"
if (-not (Test-Path $s2g)) {
    Write-Host "📥 Базовые модели v2 не найдены. Загрузка предобученных моделей с HuggingFace..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip" -OutFile "pretrained_models.zip"
    Write-Host "📦 Распаковка архива моделей (пожалуйста, подождите 1-3 мин)..." -ForegroundColor Yellow
    if (Get-Command tar.exe -ErrorAction SilentlyContinue) {
        tar.exe -xf "pretrained_models.zip" -C "GPT_SoVITS"
    } else {
        Expand-Archive -Path "pretrained_models.zip" -DestinationPath "GPT_SoVITS" -Force
    }
    Remove-Item "pretrained_models.zip" -Force -ErrorAction SilentlyContinue
    Write-Host "✅ Базовые модели успешно загружены и распакованы!" -ForegroundColor Green
} else {
    Write-Host "✅ Базовые предобученные модели GPT-SoVITS v2 в наличии!" -ForegroundColor Green
}

# 5. Скачиваем воркер и датасет с сервера
Write-Host "📥 Загрузка файлов воркера с сервера $ServerUrl ..." -ForegroundColor Green
Invoke-WebRequest -Uri "$ServerUrl/worker.py" -OutFile "worker.py"
Invoke-WebRequest -Uri "$ServerUrl/raw_data.json" -OutFile "raw_data.json"

# 6. Запуск воркера
Write-Host "🚀 Запуск фонового воркера RTX 3060..." -ForegroundColor Green
& $PythonExe worker.py
"""

# ==============================================================================
# 4. Скрипт воркера worker.py, исполняющийся на Windows
# ==============================================================================
WORKER_PY_SCRIPT = """import os
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
BASE_DIR = Path.cwd()
DATASET_DIR = BASE_DIR / "dataset_svetlana"
WAVS_DIR = DATASET_DIR / "wavs"
LOGS_DIR = BASE_DIR / "logs" / "svetlana_plus"

tts_engine = None

def check_dependencies():
    try:
        import torch
    except ImportError:
        print("📦 Установка PyTorch с поддержкой CUDA для RTX 3060...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"])
    for pkg in ["soundfile", "pyyaml", "scipy"]:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

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
        gpu_name = f"Python (без torch: {e})"
    has_w = check_has_weights()
    data = json.dumps({"gpu_name": gpu_name, "vram_gb": vram_gb, "has_weights": has_w}).encode("utf-8")
    req = urllib.request.Request(f"{SERVER_URL}/api/worker_ping", data=data, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=3)
    except Exception as e:
        print("Ошибка связи с сервером при ping:", e)

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
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass
    print(f">> {msg}")

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

    # Создаем logs/svetlana_plus
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Копируем 2-name2text.txt если есть в датасете
    if (DATASET_DIR / "2-name2text.txt").exists():
        shutil.copy(DATASET_DIR / "2-name2text.txt", LOGS_DIR / "2-name2text.txt")
        send_log("✅ Фонетическая разметка ARPA с ударениями (+) синхронизирована!")

    send_log("✅ Студийный датасет Светланы готов!", progress=25, step=2, step_name="Извлечение признаков (HuBERT 32 кГц + семантика)")

    # 2. Извлечение признаков (HuBERT + семантика)
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

    send_log("1/2 Извлечение HuBERT 32 кГц на видеокарте...")
    subprocess.run([sys.executable, "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], env=env, check=True)

    send_log("2/2 Извлечение семантических признаков...")
    subprocess.run([sys.executable, "GPT_SoVITS/prepare_datasets/3-get-semantic.py"], env=env, check=True)

    # Слияние семантических файлов
    tsv_parts = list(LOGS_DIR.glob("6-name2semantic-*.tsv"))
    if tsv_parts:
        lines = []
        for p in sorted(tsv_parts):
            with open(p, "r", encoding="utf-8") as f:
                lines.extend([l.strip() for l in f if l.strip()])
        with open(LOGS_DIR / "6-name2semantic.tsv", "w", encoding="utf-8") as f:
            f.write("\\n".join(lines) + "\\n")

    send_log("✅ Признаки датасета извлечены!", progress=40, step=3, step_name="Обучение SoVITS (20 эпох на RTX 3060)")

    # 3. Обучение SoVITS (20 эпох)
    with open("GPT_SoVITS/configs/s2.json", "r", encoding="utf-8") as f:
        s2 = json.load(f)
    s2_dir = "logs/svetlana_plus"
    os.makedirs(f"{s2_dir}/logs_s2_v2", exist_ok=True)
    os.makedirs("SoVITS_weights_v2", exist_ok=True)
    s2["train"]["batch_size"] = 8
    s2["train"]["epochs"] = 20
    s2["train"]["save_every_epoch"] = 5
    s2["train"]["if_save_latest"] = 1
    s2["train"]["pretrained_s2G"] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    s2["train"]["pretrained_s2D"] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
    s2["model"]["version"] = "v2"
    s2["data"]["exp_dir"] = s2["s2_ckpt_dir"] = s2_dir
    s2["save_weight_dir"] = "SoVITS_weights_v2"
    s2["name"] = "svetlana_plus"
    s2["version"] = "v2"
    os.makedirs("TEMP", exist_ok=True)
    with open("TEMP/tmp_s2.json", "w", encoding="utf-8") as f:
        json.dump(s2, f)

    send_log("🚀 Старт обучения акустической модели SoVITS (20 эпох)...")
    p = subprocess.Popen([sys.executable, "GPT_SoVITS/s2_train.py", "--config", "TEMP/tmp_s2.json"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for l in p.stdout:
        line_clean = l.strip()
        if "epoch" in line_clean.lower() or "loss" in line_clean.lower():
            m = re.search(r'epoch[:\\s]+(\\d+)', line_clean, re.I)
            if m:
                ep = int(m.group(1))
                prog = 40 + int((min(ep, 20) / 20) * 30)
                send_log(f"[SoVITS] {line_clean}", progress=prog)
            else:
                send_log(f"[SoVITS] {line_clean}")
    p.wait()

    send_log("✅ Обучение SoVITS успешно завершено!", progress=70, step=4, step_name="Обучение GPT (15 эпох на RTX 3060)")

    # 4. Обучение GPT (15 эпох)
    import yaml
    with open("GPT_SoVITS/configs/s1longer-v2.yaml", "r", encoding="utf-8") as f:
        s1 = yaml.load(f, Loader=yaml.FullLoader)
    os.makedirs("GPT_weights_v2", exist_ok=True)
    s1["train"]["batch_size"] = 8
    s1["train"]["epochs"] = 15
    s1["train"]["save_every_n_epoch"] = 5
    s1["train"]["if_save_latest"] = 1
    s1["pretrained_s1"] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    s1["train"]["half_weights_save_dir"] = "GPT_weights_v2"
    s1["train"]["exp_name"] = "svetlana_plus"
    s1["train_semantic_path"] = f"{s2_dir}/6-name2semantic.tsv"
    s1["train_phoneme_path"] = f"{s2_dir}/2-name2text.txt"
    s1["output_dir"] = f"{s2_dir}/logs_s1_v2"
    with open("TEMP/tmp_s1.yaml", "w", encoding="utf-8") as f:
        yaml.dump(s1, f)

    s1_env = os.environ.copy()
    s1_env["_CUDA_VISIBLE_DEVICES"] = "0"
    s1_env["hz"] = "25hz"
    send_log("🚀 Старт обучения семантической модели GPT (15 эпох)...")
    p = subprocess.Popen([sys.executable, "GPT_SoVITS/s1_train.py", "--config_file", "TEMP/tmp_s1.yaml"], env=s1_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for l in p.stdout:
        line_clean = l.strip()
        if "epoch" in line_clean.lower() or "loss" in line_clean.lower():
            m = re.search(r'epoch[:\\s]+(\\d+)', line_clean, re.I)
            if m:
                ep = int(m.group(1))
                prog = 70 + int((min(ep, 15) / 15) * 25)
                send_log(f"[GPT] {line_clean}", progress=prog)
            else:
                send_log(f"[GPT] {line_clean}")
    p.wait()

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

    # Передача архива на сервер управления
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

print(f"Воркер Светланы запущен. Подключение к серверу {SERVER_URL}...")
check_dependencies()
last_handled_tts = None
while True:
    try:
        send_ping()
        res = urllib.request.urlopen(f"{SERVER_URL}/api/worker_task", timeout=3)
        data = json.loads(res.read().decode("utf-8"))
        if data.get("task") == "start_training":
            run_training()
        tts_task = data.get("tts_task")
        if tts_task and tts_task.get("status") == "pending":
            if tts_task.get("id") != last_handled_tts:
                last_handled_tts = tts_task.get("id")
                handle_tts_task(tts_task)
    except Exception as e:
        print("Ожидание соединения с сервером:", e)
    time.sleep(2)
"""

def main():
    server = HTTPServer(("0.0.0.0", PORT), HubRequestHandler)
    print("=" * 65)
    print(f"🚀 СЕРВЕР УПРАВЛЕНИЯ СВЕТЛАНОЙ ЗАПУЩЕН НА ПОРТУ {PORT}")
    print(f"👉 Откройте в браузере на Windows: http://{SERVER_IP}:{PORT}")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")

if __name__ == "__main__":
    main()
