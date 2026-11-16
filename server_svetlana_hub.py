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
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
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

weights_exist = (BASE_DIR / "svetlana_custom_weights.zip").exists()
hub_state = {
    "worker_connected": False,
    "last_ping": 0,
    "worker_ip": "",
    "gpu_name": "Ожидание подключения Windows...",
    "vram_gb": 0.0,
    "has_weights": weights_exist,
    "status": "completed" if weights_exist else "waiting_worker",
    "step": 5 if weights_exist else 0,
    "step_name": "🎉 Модель Светланы обучена и готова к озвучке!" if weights_exist else "Ожидание подключения Windows RTX 3060",
    "progress": 100 if weights_exist else 0,
    "task": "none",              # none, start_training
    "tts_task": None,            # {"id": str, "text": str, "status": "pending"|"processing"|"ready"|"error"}
    "logs": [f"[{time.strftime('%H:%M:%S')}] 🌐 Сервер запущен. Откройте http://{SERVER_IP}:{PORT} на компьютере с Windows."],
    "elapsed_sec": 0,
    "start_time": None
}
if weights_exist:
    hub_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] 🎉 Веса модели Светланы в наличии (svetlana_custom_weights.zip). Интерактивная озвучка активна!")

class HubRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def get_server_host(self):
        hdr = self.headers.get("Host", "")
        if hdr:
            return hdr.split(":")[0]
        return get_current_lan_ip()

    def send_bytes_response(self, content_bytes, content_type="text/html; charset=utf-8", status=200, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(content_bytes)

    def send_json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_bytes_response(body, content_type="application/json; charset=utf-8", status=status)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        server_ip = self.get_server_host()

        if path == "/" or path == "/index.html":
            body = HTML_PAGE.replace("{{SERVER_IP}}", server_ip).replace("{{PORT}}", str(PORT)).encode("utf-8")
            self.send_bytes_response(body, content_type="text/html; charset=utf-8")

        elif path == "/api/status":
            now = time.time()
            if hub_state.get("last_ping", 0) > 0 and (now - hub_state["last_ping"]) > 15 and hub_state["status"] != "running":
                hub_state["worker_connected"] = False
            if hub_state["start_time"] and hub_state["status"] == "running":
                hub_state["elapsed_sec"] = int(time.time() - hub_state["start_time"])
            self.send_json_response(hub_state)

        elif path == "/api/worker_task":
            resp = {
                "task": hub_state["task"],
                "tts_task": hub_state["tts_task"]
            }
            self.send_json_response(resp)

        elif path == "/api/tts_status":
            req_id = query.get("id", [""])[0]
            status = "none"
            if hub_state["tts_task"] and hub_state["tts_task"]["id"] == req_id:
                status = hub_state["tts_task"]["status"]
            self.send_json_response({"id": req_id, "status": status})

        elif path == "/api/test_audio":
            req_id = query.get("id", [""])[0]
            audio_path = AUDIO_CACHE_DIR / f"test_{req_id}.wav"
            if audio_path.exists():
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                self.send_bytes_response(audio_bytes, content_type="audio/wav")
            else:
                self.send_response(404)
                self.send_header("Connection", "close")
                self.end_headers()

        elif path == "/raw_data.json":
            raw_path = BASE_DIR / "raw_data.json"
            if raw_path.exists():
                with open(raw_path, "rb") as f:
                    data_bytes = f.read()
                self.send_bytes_response(data_bytes, content_type="application/json; charset=utf-8")
            else:
                self.send_response(404)
                self.send_header("Connection", "close")
                self.end_headers()

        elif path == "/dataset_svetlana.zip":
            zip_p = BASE_DIR / "dataset_svetlana.zip"
            if zip_p.exists():
                with open(zip_p, "rb") as f:
                    zip_bytes = f.read()
                extra = {"Content-Disposition": 'attachment; filename="dataset_svetlana.zip"'}
                self.send_bytes_response(zip_bytes, content_type="application/zip", extra_headers=extra)
            else:
                self.send_response(404)
                self.send_header("Connection", "close")
                self.end_headers()

        elif path == "/setup.bat":
            body = SETUP_BAT_SCRIPT.replace("{{SERVER_IP}}", server_ip).replace("{{PORT}}", str(PORT)).encode("utf-8")
            extra = {"Content-Disposition": 'attachment; filename="setup_svetlana.bat"'}
            self.send_bytes_response(body, content_type="application/x-bat", extra_headers=extra)

        elif path == "/setup.ps1":
            content = (BASE_DIR / "setup.ps1").read_text(encoding="utf-8")
            body = content.replace("{{SERVER_IP}}", server_ip).replace("{{PORT}}", str(PORT)).encode("utf-8")
            self.send_bytes_response(body, content_type="text/plain; charset=utf-8")

        elif path == "/worker.py":
            content = (BASE_DIR / "worker.py").read_text(encoding="utf-8")
            body = content.replace("{{SERVER_IP}}", server_ip).replace("{{PORT}}", str(PORT)).encode("utf-8")
            self.send_bytes_response(body, content_type="text/plain; charset=utf-8")

        elif path == "/download_weights":
            weights_zip = BASE_DIR / "svetlana_custom_weights.zip"
            if weights_zip.exists():
                with open(weights_zip, "rb") as f:
                    zip_bytes = f.read()
                extra = {"Content-Disposition": 'attachment; filename="svetlana_custom_weights.zip"'}
                self.send_bytes_response(zip_bytes, content_type="application/zip", extra_headers=extra)
            else:
                self.send_response(404)
                self.send_header("Connection", "close")
                self.end_headers()
        else:
            self.send_response(404)
            self.send_header("Connection", "close")
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
            self.send_json_response({"status": "received"})
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
            self.send_json_response({"status": "saved"})
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
            self.send_json_response({"status": "ok"})

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
            self.send_json_response({"status": "queued", "id": req_id})

        elif path == "/api/clear_task":
            hub_state["task"] = "none"
            self.send_json_response({"status": "cleared"})

        elif path == "/api/worker_ping":
            # Воркер с Windows сообщает о своем статусе
            hub_state["last_ping"] = time.time()
            hub_state["worker_connected"] = True
            hub_state["worker_ip"] = self.client_address[0]
            if "gpu_name" in data and data["gpu_name"]:
                hub_state["gpu_name"] = data["gpu_name"]
            if "vram_gb" in data:
                hub_state["vram_gb"] = data["vram_gb"]
            if "has_weights" in data:
                hub_state["has_weights"] = bool(data["has_weights"])
            if hub_state["has_weights"] and hub_state["status"] != "running":
                hub_state["status"] = "completed"
                hub_state["step"] = 5
                hub_state["progress"] = 100
                hub_state["step_name"] = "🎉 Модели Светланы обучены и готовы к озвучке!"
            elif hub_state["status"] in ["waiting_worker", "error"]:
                hub_state["status"] = "ready"
                hub_state["step_name"] = "Окружение Windows готово к обучению"
                log_msg(f"🟢 Подключен компьютер Windows ({hub_state['gpu_name']}, {hub_state['vram_gb']} GB VRAM). Готов к обучению!")
            self.send_json_response({"status": "ack"})

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
            self.send_json_response({"status": "ack"})

        elif path == "/api/worker_finish":
            # Завершение обучения
            hub_state["status"] = "completed"
            hub_state["has_weights"] = True
            hub_state["task"] = "none"
            hub_state["progress"] = 100
            hub_state["step"] = 5
            hub_state["step_name"] = "Обучение успешно завершено!"
            self.send_json_response({"status": "acknowledged"})
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
      user-select: text;
      -webkit-user-select: text;
      -moz-user-select: text;
      cursor: text;
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
      <div style="font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center;">
        <span>Терминал вывода RTX 3060 в реальном времени:</span>
        <div style="display: flex; gap: 8px; align-items: center;">
          <button onclick="copyTerminalLogs()" style="font-size: 11px; padding: 4px 10px; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.4); color: #60a5fa; border-radius: 6px; cursor: pointer; transition: all 0.2s;" onmouseover="this.style.background='rgba(59, 130, 246, 0.3)'" onmouseout="this.style.background='rgba(59, 130, 246, 0.15)'">📋 Скопировать весь лог</button>
          <span id="timerText" style="font-family: monospace;">00:00</span>
        </div>
      </div>
      <div id="terminal" class="terminal">Ожидание подключения воркера Windows...</div>
    </div>
  </div>

  <script>
    function copyTerminalLogs() {
      const term = document.getElementById('terminal');
      const text = term.innerText;
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
          alert('Логи терминала успешно скопированы в буфер обмена!');
        }).catch(() => fallbackCopy(text));
      } else {
        fallbackCopy(text);
      }
    }

    function fallbackCopy(text) {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.left = '-9999px';
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand('copy');
        alert('Логи терминала успешно скопированы в буфер обмена!');
      } catch (e) {
        alert('Не удалось автоматически скопировать. Выделите текст мышью и нажмите Ctrl+C.');
      }
      document.body.removeChild(ta);
    }

    function copyPS() {
      const code = document.getElementById('psCode').innerText;
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(code).then(() => {
          alert('Команда скопирована в буфер обмена! Нажмите Ctrl+V в окне PowerShell на Windows.');
        }).catch(() => fallbackCopy(code));
      } else {
        fallbackCopy(code);
      }
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
        const newLogs = (data.logs || []).join('\n');
        if (term.innerText !== newLogs) {
          const sel = window.getSelection();
          const isUserSelecting = sel && sel.rangeCount > 0 && term.contains(sel.anchorNode) && !sel.isCollapsed;
          if (!isUserSelecting) {
            const isScrolledToBottom = (term.scrollHeight - term.clientHeight) <= (term.scrollTop + 40);
            term.innerText = newLogs;
            if (isScrolledToBottom) {
              term.scrollTop = term.scrollHeight;
            }
          }
        }

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
def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), HubRequestHandler)
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
