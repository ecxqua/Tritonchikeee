from pathlib import Path
import gdown

def download_models_folder() -> None:
    bot_dir = Path(__file__).resolve().parents[0]
    models_dir = bot_dir / "models"
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

    print(f"Загружет файлы в {models_dir.resolve()}")
    gdown.download_folder(
        url=folder_url,
        output=str(models_dir),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    print("Загрузка весов завершена")


if __name__ == "__main__":
    download_models_folder()
