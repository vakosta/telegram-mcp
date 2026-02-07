FROM python:3.12-slim

# Установка Node.js (для Supergateway)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем и устанавливаем Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код сервера
COPY . .

# Порт для SSE (Railway назначит через переменную PORT)
ENV PORT=8000
EXPOSE 8000

# Запускаем telegram-mcp через Supergateway (stdio → SSE)
CMD npx -y supergateway \
    --stdio "python main.py" \
    --port ${PORT}
```

### 4. Деплой на Railway

1. Зайдите на [railway.app](https://railway.app) и создайте новый проект
2. Выберите **Deploy from GitHub repo** → подключите свой форк
3. Railway автоматически обнаружит `Dockerfile` и начнёт сборку

### 5. Настройте переменные окружения

В настройках сервиса на Railway (вкладка **Variables**) добавьте:

| Переменная | Значение |
|---|---|
| `TELEGRAM_API_ID` | ваш API ID |
| `TELEGRAM_API_HASH` | ваш API Hash |
| `TELEGRAM_SESSION_STRING` | строка сессии из шага 2 |
| `PORT` | `8000` (или Railway назначит автоматически) |

### 6. Откройте публичный домен

В разделе **Settings → Networking** Railway сервиса нажмите **Generate Domain**. Вы получите URL вроде:
```
https://telegram-mcp-production-xxxx.up.railway.app
