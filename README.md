# 🎙️ Дообучение студийного голоса Светланы (ru-RU-SvetlanaNeural)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeonidYasin/colab_svetlana_finetune/blob/main/colab_svetlana_finetune.ipynb)

Проект для тонкого дообучения (Fine-Tuning) нейросетевого голоса **Светланы** на базе архитектуры **GPT-SoVITS v2** с полным и детерминированным контролем ударений знаками плюса (`+`).

Поддерживается запуск:
- 💻 **Локально под Windows на видеокарте NVIDIA (RTX 3060 / 4060 / 3070 и др.)** — всего ~7–10 минут, без лимитов и очередей!
- ☁️ **В облаке Google Colab (GPU Tesla T4)** — в 1 клик через браузер.

---

## 💻 Быстрый запуск под Windows (на локальной RTX 3060)

### Вариант 1. Если у вас уже скачана официальная сборка GPT-SoVITS для Windows
1. Скачайте файлы из этого репозитория: `train_svetlana_windows.py`, `train_svetlana.bat` и `raw_data.json`.
2. Положите их прямо в папку `GPT-SoVITS` (где находятся файлы `webui.py` и папка `runtime`).
3. Запустите двойным кликом **`train_svetlana.bat`**.
4. Скрипт сам:
   - Проверит видеокарту RTX 3060.
   - Сгенерирует студийный датасет Светланы (102 аудиоклипа с ударениями `+`).
   - Извлечёт признаки (HuBERT + Семантика).
   - Обучит модель **SoVITS (20 эпох)** за ~4 минуты.
   - Обучит модель **GPT (15 эпох)** за ~3 минуты.
   - Сохранит готовые веса и упакует их в `svetlana_custom_weights.zip`.

### Вариант 2. Автоматическая установка через PowerShell
Откройте PowerShell в любой папке и выполните:
```powershell
git clone https://github.com/LeonidYasin/colab_svetlana_finetune.git
cd colab_svetlana_finetune
powershell -ExecutionPolicy Bypass -File setup_and_train_windows.ps1
```

---

## ☁️ Запуск в Google Colab (в 1 клик)

👉 **[Открыть блокнот в Google Colab](https://colab.research.google.com/github/LeonidYasin/colab_svetlana_finetune/blob/main/colab_svetlana_finetune.ipynb)**

1. В верхнем меню выберите **«Среда выполнения»** ➔ **«Выполнить все»**.
2. Вся процедура занимает ~12–14 минут на бесплатном GPU T4.

---

## 🌟 Ключевые возможности
* **Точный контроль омографов и ударений (+):**
  * `вдов+ы` (род. падеж) vs `вд+овы` (мн. число)
  * `перво+образ` (корень «-о́браз»)
  * `стор+ицей` (литературная норма)
  * `в+ерхом` vs `верх+ом`
  * `обр+ёк`, `опекун+ам`, `тяжелобольн+ом`
* **Zero-Upload:** Датасет создаётся автоматически в оперативной памяти/на диске, без необходимости загружать тяжелые архивы.
