import yaml


# Загрузка конфигураций
def load_config(config_path="config/config.yaml"):
    """Загрузка файла конфигураций 'config.yaml'"""
    try:
        with open(config_path, "r", encoding="UTF-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Добавьте файл конфигурации для анализа: config/config.yaml"
        )
