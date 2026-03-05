# ReID_newts

Веса обученных моделей по ссылке - https://drive.google.com/drive/folders/1UsnulLQ6BuiWZvuEO2ozhSZSGt78Hc8a

## Настройка бота

1. Создайте файл `.env` в корневой директории
2. Добавьте в него ваш токен бота: TOKEN=YOUR_TOKEN
3. Переместите скачанные модели в директорию bot/models
4. Создать в папки conservation и results в bot/ (bot/conservation и bot/results)

## Сборка и запуск через Docker
### Сборка образа
```bash
docker build -t oneomebot .
```
### Запуск контейнера
```bash
docker run -d --name oneomebot -e TOKEN="YOUR_TOKEN" oneomebot
```
