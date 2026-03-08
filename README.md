# ReID_newts

`ReID_newts` — это Telegram-бот для реидентификации тритонов по фото брюшка.

## Запуск telegram-бота
1. Зарегистрируйте бота в BotsFather.
2.  Создайте файл `.env` в корневой директории. Добавьте в него ваш токен бота: TOKEN=YOUR_TOKEN
3. Скайчате веса обученных моделей и модели: https://drive.google.com/drive/folders/1UsnulLQ6BuiWZvuEO2ozhSZSGt78Hc8a
4. Переместите скачанные файлы в директорию `bot/models`.
5. Создайте пустые папки `bot/conservation` и `bot/results`.
7. Подтяните все зависимости из requirements.txt.
8. Запустите проект в режиме python-модуля:
```
python -m bot.app
```

## Сборка и запуск через Docker
### Сборка образа
```bash
docker build -t oneomebot .
```
### Запуск контейнера
```bash
docker run -d --name oneomebot -e TOKEN="YOUR_TOKEN" oneomebot
```