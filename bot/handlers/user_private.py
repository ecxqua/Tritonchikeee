from aiogram import Router, types, F, Bot
from aiogram.filters import CommandStart
from bot.handlers.model_utils import photo_processing
import os
from pathlib import Path

# Роутер для обработки сообщений в приватных чатах с ботом
user_private_router = Router()


@user_private_router.message(CommandStart())
async def start(message: types.Message):
    """Обработчик команды /start - приветственное сообщение"""
    await message.answer(
        'Данный бот создан для демонстрации возможностей модели.\n'
        'Пожалуйста, загрузите фотографию брюшка тритона, '
        'и модель выведет вам 5 самых похожих изображений\n\n'
        'Отправляйте изображение как файл (Document), '
        'чтобы Telegram не сжимал фото.\n'
        'Поддерживаемые форматы: JPG, PNG, JPEG'
    )


@user_private_router.message(F.photo)
async def handle_photo(message: types.Message, bot: Bot):
    """
    Обработчик загрузки фотографий (через обычное фото в Telegram)
    """
    try:
        # Получаем фото с наивысшим качеством (последний элемент в списке)
        photo = message.photo[-1]
        # Получаем информацию о файле
        photo_file = await bot.get_file(photo.file_id)

        # Формируем путь для сохранения файла
        filename = "image.jpg"
        filepath = os.path.join(bot.save_dir, filename)

        # Скачиваем файл на сервер
        await bot.download_file(photo_file.file_path, filepath)

        # Уведомляем пользователя о сохранении
        await message.answer(
            '✅ Ваша фотография была успешно сохранена!\n'
            '⌛ Пожалуйста, подождите, пока завершится работа алгоритма'
        )

        # Обрабатываем изображение с помощью модели
        success = await photo_processing(bot, filepath)

        if success:
            await send_results(message, bot)
        else:
            await message.answer("❌ Ошибка при обработке изображения")

    except Exception as e:
        # Обрабатываем любые исключения и уведомляем пользователя
        await message.answer(f"❌ Ошибка при обработке фото: {str(e)}")


@user_private_router.message(F.document)
async def handle_document(message: types.Message, bot: Bot):
    """
    Обработчик загрузки файлов (изображений как документов)
    """
    try:
        # Проверяем, что это изображение по расширению файла
        document = message.document

        # Получаем расширение файла
        file_ext = Path(document.file_name).suffix.lower() if document.file_name else ''

        # Разрешенные форматы изображений
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

        # Проверяем MIME тип или расширение файла
        is_image = (
                document.mime_type and document.mime_type.startswith('image/') or
                file_ext in allowed_extensions
        )

        if not is_image:
            await message.answer(
                "❌ Пожалуйста, отправьте изображение в формате JPG, PNG, JPEG или другой поддерживаемый формат.\n"
                "Поддерживаемые форматы: JPG, JPEG, PNG, BMP, GIF, WEBP"
            )
            return

        # Получаем информацию о файле
        document_file = await bot.get_file(document.file_id)

        # Определяем расширение для сохранения файла
        if file_ext and file_ext in allowed_extensions:
            filename = f"image{file_ext}"
        else:
            # Если расширение не распознано, используем jpg по умолчанию
            filename = "image.jpg"

        filepath = os.path.join(bot.save_dir, filename)

        # Скачиваем файл на сервер
        await bot.download_file(document_file.file_path, filepath)

        # Уведомляем пользователя о сохранении
        await message.answer(
            f'✅ Ваш файл "{document.file_name}" был успешно сохранен!\n'
            '⌛ Пожалуйста, подождите, пока завершится работа алгоритма'
        )

        # Обрабатываем изображение с помощью модели
        success = await photo_processing(bot, filepath)

        if success:
            await send_results(message, bot)
        else:
            await message.answer("❌ Ошибка при обработке изображения")

    except Exception as e:
        # Обрабатываем любые исключения и уведомляем пользователя
        await message.answer(f"❌ Ошибка при обработке файла: {str(e)}")


async def send_results(message: types.Message, bot: Bot):
    """
    Функция для отправки результатов обработки
    """
    try:
        # Формируем список путей к результатам обработки
        result_files = [
            os.path.join(bot.result_dir, f"top{i}.jpg")
            for i in range(1, bot.size_answer + 1)
        ]

        # Фильтруем только существующие файлы
        existing_files = [f for f in result_files if os.path.exists(f)]

        # Проверяем, что есть результаты обработки
        if not existing_files:
            await message.answer("❌ Результаты обработки не найдены")
            return

        # Проверяем наличие файла с текстовыми результатами
        res_file_path = os.path.join(bot.result_dir, "res.txt")
        if os.path.exists(res_file_path):
            # Читаем текстовые результаты из файла
            with open(res_file_path, 'r', encoding='utf-8') as f:
                res_str = f.read()

            # Создаем медиагруппу для отправки нескольких фото
            media_group = []
            for i, filepath in enumerate(existing_files[:bot.size_answer]):
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        # Добавляем подпись только к первому изображению
                        caption = f"Результаты обработки:\n\n{res_str}" if i == 0 else None
                        media_group.append(types.InputMediaPhoto(
                            media=types.BufferedInputFile(
                                f.read(),
                                filename=f"top{i + 1}.jpg"
                            ),
                            caption=caption
                        ))

            # Отправляем медиагруппу с результатами
            if media_group:
                await message.answer_media_group(media=media_group)
                await message.answer(f"✅ Готово! Вот {len(media_group)} самых похожих изображений")
            else:
                await message.answer("❌ Не удалось загрузить результаты обработки")
        else:
            await message.answer("❌ Файл с результатами не найден")

    except Exception as e:
        # Обрабатываем любые исключения и уведомляем пользователя
        await message.answer(f"❌ Ошибка при отправке результатов: {str(e)}")