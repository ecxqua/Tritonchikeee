import asyncio
import os
from pathlib import Path
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from bot.handlers.user_private import user_private_router
from dotenv import find_dotenv, load_dotenv
import yaml
import gdown

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / '.env'
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)
else:
    load_dotenv(find_dotenv(), override=True)

token = (os.getenv("TOKEN") or "").strip()
if not token or token == "PASTE_YOUR_TELEGRAM_BOT_TOKEN_HERE":
    raise RuntimeError("TOKEN is not set in .env")


# Загрузка конфигураций
def load_config(config_path="config/config.yaml"):
    """Загрузка файла конфигураций 'config.yaml'"""
    try:
        with open(config_path, "r", encoding="UTF-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Добавьте файл конфигурации для анализа: config/config.yaml")

# Функция загрузки папки весов из Google Drive
def download_models_folder():
    models_dir = Path("models")
    required_files = [
        models_dir / "best_seg.pt",
        models_dir / "best_model.pth",
    ]
    folder_url = (
        "https://drive.google.com/drive/folders/"
        "1OjB1VAS6FyROYeWpnmkwlt80AOPEq9_7?usp=drive_link"
    )

    models_dir.mkdir(exist_ok=True)

    if all(path.exists() for path in required_files):
        print("Файлы весов уже есть")
        return

    print(f"Загружаю файлы в {models_dir.resolve()}")

    try:
        gdown.download_folder(
            url=folder_url,
            output=str(models_dir),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
        print("Загрузка весов завершена")
    except Exception as e:
        print(f"Ошибка при загрузке весов: {e}")
        raise

bot = Bot(token=token, default=DefaultBotProperties())
bot.config = load_config()
# bot.save_dir = 'bot/conservation'
# bot.result_dir = 'bot/results'
# bot.size_answer = 5

dp = Dispatcher()
dp.include_router(user_private_router)

async def on_startup():
    print("bot is started")


async def on_shutdown():
    print("bot is died")

async def main():
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    download_models_folder()
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())