import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import map_coordinates, gaussian_filter1d


class TritonMaskUnwrapper:
    def __init__(self, seg_model_path="yolo11s_seg_23_12/weights/best.pt"):
        """
        Инициализация модели сегментации для развертки брюшка тритона
        """
        self.seg_model = YOLO(seg_model_path)

        # Параметры для настройки (должны совпадать с deployment_scripts/deployment_yolo.py)
        self.TRIM_TOP_PCT = 0.3   # 30% сверху
        self.TRIM_BOTTOM_PCT = 0.15  # 15% снизу
        self.FINAL_SIZE = 244  # Должен совпадать с тем, чем готовился датасет

    def extract_smooth_centerline(self, mask, step=2, sigma_x=3):
        """
        Строит медиальную линию по маске.
        """
        h, w = mask.shape
        ys = np.arange(0, h, step)
        pts = []

        for y in ys:
            xs = np.where(mask[y] > 0)[0]
            if len(xs) == 0:
                continue
            x_center = 0.5 * (xs.min() + xs.max())
            pts.append([x_center, y])

        if len(pts) < 2:
            raise ValueError("Центральная линия не найдена")

        centerline = np.array(pts, dtype=np.float32)
        centerline[:, 0] = gaussian_filter1d(centerline[:, 0], sigma=sigma_x)
        return centerline

    def get_segmentation_mask(self, image, img_path):
        """
        Получает маску сегментации с помощью YOLO модели.
        """
        h_img, w_img = image.shape[:2]
        seg_results = self.seg_model(img_path)
        masks = seg_results[0].masks
        if masks is None:
            return None

        mask_tensor = masks.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask_tensor, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        mask = (mask_resized * 255).astype(np.uint8)
        return mask

    def unwrap_belly_trimmed_ends(self, image, mask, centerline_raw, save_path):
        """
        Разворачивает изображение брюшка тритона только по маске.
        """
        trim_top_pct = self.TRIM_TOP_PCT
        trim_bottom_pct = self.TRIM_BOTTOM_PCT
        final_size = self.FINAL_SIZE

        if len(centerline_raw) < 2:
            raise ValueError("Центральная линия слишком короткая")

        # 1. Ресэмплинг в параметрическом пространстве
        x = centerline_raw[:, 0]
        y = centerline_raw[:, 1]
        t = np.linspace(0, 1, len(centerline_raw))
        t_new = np.linspace(0, 1, final_size)

        x_interp = np.interp(t_new, t, x)
        y_interp = np.interp(t_new, t, y)

        x_smooth = gaussian_filter1d(x_interp, sigma=7)
        y_smooth = gaussian_filter1d(y_interp, sigma=3)
        centerline_smooth = np.column_stack((x_smooth, y_smooth)).astype(np.float32)

        # 2. Обрезка по процентам
        n = len(centerline_smooth)
        top_cut = int(n * trim_top_pct)
        bot_cut = int(n * trim_bottom_pct)

        if top_cut + bot_cut >= n - 2:
            raise ValueError("Слишком агрессивный trim%, центрлайн почти исчез")

        centerline_trimmed = centerline_smooth[top_cut: n - bot_cut]
        n = len(centerline_trimmed)

        # 3. Выпрямление концов
        if n >= 10:
            k_tail = max(3, int(0.05 * n))

            x_top, y_top = centerline_trimmed[k_tail, 0], centerline_trimmed[k_tail, 1]
            x_bot, y_bot = centerline_trimmed[n - k_tail - 1, 0], centerline_trimmed[n - k_tail - 1, 1]

            def x_on_midline(yv):
                if y_bot == y_top:
                    return x_top
                t_rel = (yv - y_top) / (y_bot - y_top)
                return x_top + t_rel * (x_bot - x_top)

            for i in range(k_tail):
                y_i = centerline_trimmed[i, 1]
                centerline_trimmed[i, 0] = x_on_midline(y_i)

            for i in range(n - k_tail, n):
                y_i = centerline_trimmed[i, 1]
                centerline_trimmed[i, 0] = x_on_midline(y_i)

        # 4. Вычисление нормалей
        dx = gaussian_filter1d(np.gradient(centerline_trimmed[:, 0]), sigma=3)
        dy = gaussian_filter1d(np.gradient(centerline_trimmed[:, 1]), sigma=3)
        lengths = np.hypot(dx, dy) + 1e-6
        normals = np.column_stack((-dy / lengths, dx / lengths))

        h_img, w_img = image.shape[:2]
        lines = []
        max_strip_width = 0

        num_points = len(centerline_trimmed)

        # 5. Извлечение полос перпендикулярно центральной линии
        for i in range(num_points):
            cx, cy = centerline_trimmed[i]
            nx, ny = normals[i]

            length_neg, length_pos = 0, 0

            # Отрицательное направление
            for step in range(1, max(h_img, w_img)):
                px, py = int(cx - nx * step), int(cy - ny * step)
                if not (0 <= px < w_img and 0 <= py < h_img) or mask[py, px] == 0:
                    break
                length_neg += 1

            # Положительное направление
            for step in range(1, max(h_img, w_img)):
                px, py = int(cx + nx * step), int(cy + ny * step)
                if not (0 <= px < w_img and 0 <= py < h_img) or mask[py, px] == 0:
                    break
                length_pos += 1

            strip_width = length_neg + length_pos
            max_strip_width = max(max_strip_width, strip_width)

            # Извлекаем пиксели вдоль нормали
            line = []
            for j in range(-length_neg, length_pos):
                px = cx + nx * j
                py = cy + ny * j
                if 0 <= int(py) < h_img and 0 <= int(px) < w_img:
                    coords_sample = np.array([[py], [px]], dtype=np.float32)
                    pixel = np.stack([
                        map_coordinates(image[:, :, c], coords_sample, order=1, mode="reflect")[0]
                        for c in range(3)
                    ], axis=-1)
                    line.append(pixel)

            if line:
                lines.append(np.array(line, dtype=np.uint8))

        if not lines:
            raise ValueError("Не удалось построить развёртку")

        # 6. Сборка развертки
        unwrapped = np.zeros((num_points, max_strip_width, 3), dtype=np.uint8)
        for i, line in enumerate(lines):
            if line.shape[0] >= 2:
                resized_line = cv2.resize(
                    line[None, :, :],
                    (max_strip_width, 1),
                    interpolation=cv2.INTER_LINEAR,
                )
                unwrapped[i] = resized_line[0]

        # 7. Ресайз до финального размера
        final = cv2.resize(unwrapped, (final_size, final_size), interpolation=cv2.INTER_LINEAR)

        # 8. Сохранение
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def save_segmentation_debug(image, mask, output_dir):
    """Сохраняет артефакты сегментации для визуальной проверки YOLO."""
    os.makedirs(output_dir, exist_ok=True)

    mask_binary = (mask > 127).astype(np.uint8) * 255
    mask_path = os.path.join(output_dir, "yolo_mask.png")
    cv2.imwrite(mask_path, mask_binary)

    masked_region = cv2.bitwise_and(image, image, mask=mask_binary)
    masked_region_path = os.path.join(output_dir, "yolo_region.jpg")
    cv2.imwrite(
        masked_region_path,
        masked_region,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )

    overlay = image.copy()
    contours, _ = cv2.findContours(
        mask_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(overlay, [largest_contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 128, 255), 2)

    overlay_path = os.path.join(output_dir, "yolo_overlay.jpg")
    cv2.imwrite(overlay_path, overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


async def process_single_image(img_path, output_dir):
    """
    Основная функция обработки одного изображения.
    Сохраняет результат как image_cropped.jpg в указанной директории.

    Args:
        img_path: путь к изображению
        output_dir: директория для сохранения результата

    Returns:
        success: True/False - успех обработки
    """
    try:
        # Валидация формата файла
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            return False

        # Загрузка изображения
        image = cv2.imread(img_path)
        if image is None:
            return False

        # Создание unwrapper
        unwrapper = TritonMaskUnwrapper(seg_model_path="yolo11s_seg_23_12/weights/best.pt")

        # Получение маски
        mask = unwrapper.get_segmentation_mask(image, img_path)
        if mask is None:
            return False

        save_segmentation_debug(image, mask, output_dir)

        # Извлечение центральной линии
        centerline = unwrapper.extract_smooth_centerline(mask)

        # Создание директории и сохранение результата
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "image_cropped.jpg")

        # Развертка брюшка
        unwrapper.unwrap_belly_trimmed_ends(image, mask, centerline, save_path)
        return True

    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {str(e)}")
        return False


# Синхронная версия для импорта
def process_single_image_sync(img_path, output_dir):
    """
    Синхронная версия функции обработки изображения.
    Импортируйте эту функцию в ваш код.
    """
    import asyncio

    # Создаем новую event loop для синхронного вызова
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(process_single_image(img_path, output_dir))