"""
pipeline/deployment_yolo_new.py — Ядро сегментации брюшка тритона (YOLO)
Версия: 3.0 (Refactored for API)

АРХИТЕКТУРНЫЕ ИЗМЕНЕНИЯ:
    1. Debug-файлы → опционально (флаг debug=False по умолчанию)
    2. Возврат numpy array → нет лишнего I/O (in-memory pipeline) (TODO: требует подключения)
    3. Возвращаемый тип → Dict вместо bool (больше информации)

Зависимости:
    - models/best_seg.pt — веса YOLO модели сегментации
"""

import os
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO
from scipy.ndimage import map_coordinates, gaussian_filter1d

logger = logging.getLogger(__name__)

# =============================================================================
# КЛАСС РАЗВЕРТКИ БРЮШКА
# =============================================================================

class TritonMaskUnwrapper:
    """
    Класс для развертки брюшка тритона по маске сегментации.
    
    Args:
        trim_top_pct: Процент обрезки сверху (по умолчанию 0.15)
        trim_bottom_pct: Процент обрезки снизу (по умолчанию 0.3)
        final_size: Финальный размер изображения (по умолчанию 244)
        seg_model_path: Путь к модели YOLO сегментации
    """
    
    def __init__(
        self,
        trim_top_pct: float = 0.15,
        trim_bottom_pct: float = 0.3,
        final_size: int = 244,
        seg_model_path: str = "models/best_seg.pt"
    ):
        # Валидация пути к модели
        if not Path(seg_model_path).exists():
            raise FileNotFoundError(f"Модель сегментации не найдена: {seg_model_path}")
        
        self.seg_model = YOLO(seg_model_path)
        
        # Параметры для настройки
        self.TRIM_TOP_PCT = trim_top_pct
        self.TRIM_BOTTOM_PCT = trim_bottom_pct
        self.FINAL_SIZE = final_size
    
    def extract_smooth_centerline(self, mask: np.ndarray, step: int = 2, sigma_x: float = 3) -> np.ndarray:
        """
        Строит медиальную линию по маске.
        
        Args:
            mask: Маска сегментации (H, W)
            step: Шаг дискретизации по Y
            sigma_x: Коэффициент сглаживания по X
        
        Returns:
            np.ndarray: Центральная линия (N, 2)
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

    def get_segmentation_mask(self, image: np.ndarray, img_path: str) -> Optional[np.ndarray]:
        """
        Получает маску сегментации с помощью YOLO модели.
        
        Args:
            image: Исходное изображение (BGR, H, W, 3)
            img_path: Путь к изображению (для YOLO)
        
        Returns:
            np.ndarray: Маска (H, W, uint8) или None если не найдена
        """
        h_img, w_img = image.shape[:2]
        seg_results = self.seg_model(img_path, verbose=False)
        
        if not seg_results or seg_results[0].masks is None:
            logger.warning(f"Маска не найдена для {img_path}")
            return None

        mask_tensor = seg_results[0].masks.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask_tensor, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        mask = (mask_resized > 0.5).astype(np.uint8) * 255

        # Оставляем только крупнейшую связную область
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        
        return mask

    def unwrap_belly_to_array(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        centerline_raw: np.ndarray
    ) -> np.ndarray:
        """
        Разворачивает изображение брюшка и возвращает numpy array (без сохранения на диск).
        
        Args:
            image: Исходное изображение (BGR, H, W, 3)
            mask: Маска сегментации (H, W)
            centerline_raw: Центральная линия (N, 2)
        
        Returns:
            np.ndarray: Развёрнутое изображение брюшка (FINAL_SIZE, FINAL_SIZE, 3)
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

        return final  # ← Возвращаем array, не сохраняем

    def unwrap_belly_trimmed_ends(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        centerline_raw: np.ndarray,
        save_path: str
    ) -> np.ndarray:
        """
        Разворачивает изображение брюшка и сохраняет на диск.
        
        Args:
            image: Исходное изображение
            mask: Маска сегментации
            centerline_raw: Центральная линия
            save_path: Путь для сохранения
        
        Returns:
            np.ndarray: Развёрнутое изображение
        """
        # Используем unwrap_belly_to_array() для вычислений
        final = self.unwrap_belly_to_array(image, mask, centerline_raw)
        
        # Сохраняем на диск
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        return final

# =============================================================================
# DEBUG-ФУНКЦИИ (ОПЦИОНАЛЬНО)
# =============================================================================

def save_segmentation_debug(image: np.ndarray, mask: np.ndarray, output_dir: str) -> None:
    """
    Сохраняет артефакты сегментации для визуальной проверки YOLO.
    
    Args:
        image: Исходное изображение
        mask: Маска сегментации
        output_dir: Директория для сохранения
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mask_binary = (mask > 127).astype(np.uint8) * 255
    mask_path = os.path.join(output_dir, "yolo_mask.png")
    cv2.imwrite(mask_path, mask_binary)

    masked_region = cv2.bitwise_and(image, image, mask=mask_binary)
    masked_region_path = os.path.join(output_dir, "yolo_region.jpg")
    cv2.imwrite(masked_region_path, masked_region, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    overlay = image.copy()
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(overlay, [largest_contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 128, 255), 2)

    overlay_path = os.path.join(output_dir, "yolo_overlay.jpg")
    cv2.imwrite(overlay_path, overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    logger.debug(f"Debug-артефакты сохранены в {output_dir}")

# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ
# =============================================================================

async def process_single_image(
    img_path: str,
    output_dir: Optional[str] = None,
    crop_name: Optional[str] = None,
    trim_top_pct: float = 0.15,
    trim_bottom_pct: float = 0.3,
    final_size: int = 244,
    seg_model_path: str = "models/best_seg.pt",
    debug: bool = False,
    return_array: bool = True,
) -> Dict[str, Any]:
    """
    Основная функция обработки одного изображения.
    
    АРХИТЕКТУРНЫЕ ИЗМЕНЕНИЯ В ВЕРСИИ 3.0:
        1. Возвращает Dict вместо bool (больше информации)
        2. Возвращает numpy array (return_array=True) — нет лишнего I/O
        3. Debug-файлы опционально (debug=False по умолчанию)
        4. Сохранение на диск только для архива (не для pipeline)
    
    Args:
        img_path: Путь к изображению
        output_dir: Директория для сохранения (если None, не сохраняем)
        trim_top_pct: Процент обрезки сверху
        trim_bottom_pct: Процент обрезки снизу
        final_size: Финальный размер изображения
        seg_model_path: Путь к модели YOLO
        debug: Сохранять ли debug-артефакты (маска, overlay)
        return_array: Возвращать ли numpy array (для in-memory pipeline)
    
    Returns:
        Dict:
            - success: bool (успех обработки)
            - crop_array: np.ndarray | None (кроп брюшка в памяти)
            - crop_path: str | None (путь к сохранённому файлу)
            - error: str | None (описание ошибки)
    """
    result: Dict[str, Any] = {
        'success': False,
        'crop_array': None,
        'crop_path': None,
        'error': None
    }
    
    try:
        # Валидация формата файла
        if not Path(img_path).suffix.lower() in [".jpg", ".jpeg", ".png"]:
            result['error'] = "Неподдерживаемый формат файла"
            return result

        # Загрузка изображения
        # эти махинации нужны, поскольку иногда!! process_single_image
        # из некоторого неведомого неотслеженного источника получает
        # img_path: Path вместо img_path: str и все ломается.
        image = cv2.imread(str(Path(img_path).resolve()))
        if image is None:
            result['error'] = "Не удалось загрузить изображение"
            return result

        # Создание unwrapper
        unwrapper = TritonMaskUnwrapper(
            trim_top_pct,
            trim_bottom_pct,
            final_size,
            seg_model_path
        )

        # Получение маски
        mask = unwrapper.get_segmentation_mask(image, img_path)
        if mask is None:
            result['error'] = "Маска не найдена (YOLO не детектировал тритона)"
            return result

        # Debug-артефакты (только если нужно)
        if debug:
            save_segmentation_debug(image, mask, output_dir or "output/debug")

        # Извлечение центральной линии
        centerline = unwrapper.extract_smooth_centerline(mask)

        # Развертка брюшка → numpy array (в памяти!)
        unwrapped = unwrapper.unwrap_belly_to_array(image, mask, centerline)
        
        if return_array:
            result['crop_array'] = unwrapped
        # Сохранение на диск для дальнейшей обработки.
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if not crop_name:
                crop_name = "image_cropped.jpg"
            else:
                crop_name = crop_name + ".jpg"
            save_path = os.path.join(output_dir, crop_name)
            cv2.imwrite(save_path, unwrapped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            result['crop_path'] = save_path
            logger.debug(f"Кроп сохранён: {save_path}")

        result['success'] = True
        logger.info(f"Обработка успешна: {Path(img_path).name}")
        return result

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Ошибка при обработке {img_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return result

# =============================================================================
# СИНХРОННАЯ ОБЁРТКА
# =============================================================================

def process_single_image_sync(
    img_path: str,
    output_dir: Optional[str] = None,
    trim_top_pct: float = 0.15,
    trim_bottom_pct: float = 0.3,
    final_size: int = 244,
    seg_model_path: str = "models/best_seg.pt",
    debug: bool = False,
    return_array: bool = True,
) -> Dict[str, Any]:
    """
    Синхронная версия process_single_image().
    
    Args:
        См. process_single_image()
    
    Returns:
        Dict: Результаты обработки
    """
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(process_single_image(
        img_path=img_path,
        output_dir=output_dir,
        trim_top_pct=trim_top_pct,
        trim_bottom_pct=trim_bottom_pct,
        final_size=final_size,
        seg_model_path=seg_model_path,
        debug=debug,
        return_array=return_array
    ))