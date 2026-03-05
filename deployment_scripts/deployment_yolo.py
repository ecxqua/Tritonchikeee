import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import map_coordinates, gaussian_filter1d

# ===============================
# КОНФИГ ПУТЕЙ И ПАРАМЕТРОВ
# ===============================

ROOT_DIR = r"C:/klasss/archive/seg"

# Папки с данными
INPUT_ROOT = r"C:/klasss/archive/seg/2"              # исходные фото по подпапкам
SEND_PHOTO_DIR = os.path.join(ROOT_DIR, "send_photo")  # конечный вывод (unwrap)

# Модель сегментации
SEG_MODEL_PATH = r"C:/klasss/archive/seg/yolo11s_seg_newts_exp1/weights/best.pt"

# Сколько процентов центрлайна отрезать сверху/снизу ПОСЛЕ ресэмплинга
TRIM_TOP_PCT = 0.3     # 30 % наверху
TRIM_BOTTOM_PCT = 0.15 # 15 % внизу

# Итоговый размер развёртки (до 224x224)
FINAL_SIZE = 244

# Включать ли дебаг-картинки (красная/зелёная ось)
DEBUG = False

os.makedirs(SEND_PHOTO_DIR, exist_ok=True)

seg_model = YOLO(SEG_MODEL_PATH)

# ===============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===============================

def extract_smooth_centerline(mask, step=2, sigma_x=3):
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


def save_debug_centerlines(image, mask, centerline_raw, centerline_final, save_path):
    debug = image.copy()

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored = np.zeros_like(mask_rgb)
    mask_colored[:, :, 1] = mask  # зелёный канал
    debug = cv2.addWeighted(debug, 0.7, mask_colored, 0.3, 0)

    # красная линия — исходный центрлайн
    if centerline_raw is not None and len(centerline_raw) > 1:
        for i in range(len(centerline_raw) - 1):
            x1, y1 = map(int, centerline_raw[i])
            x2, y2 = map(int, centerline_raw[i + 1])
            cv2.line(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # зелёная линия — обрезанный центрлайн
    if centerline_final is not None and len(centerline_final) > 1:
        for i in range(len(centerline_final) - 1):
            x1, y1 = map(int, centerline_final[i])
            x2, y2 = map(int, centerline_final[i + 1])
            cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, debug, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def unwrap_belly_mask_only(
    image,
    mask,
    centerline_raw,
    save_path="unwrapped.jpg",
    final_size=FINAL_SIZE,
    trim_top_pct=TRIM_TOP_PCT,
    trim_bottom_pct=TRIM_BOTTOM_PCT,
):
    if len(centerline_raw) < 2:
        raise ValueError("Центральная линия слишком короткая")

    # --- 1. Ресэмплинг в параметрическом пространстве t ---
    x = centerline_raw[:, 0]
    y = centerline_raw[:, 1]
    t = np.linspace(0, 1, len(centerline_raw))
    t_new = np.linspace(0, 1, final_size)

    x_interp = np.interp(t_new, t, x)
    y_interp = np.interp(t_new, t, y)

    x_smooth = gaussian_filter1d(x_interp, sigma=7)
    y_smooth = gaussian_filter1d(y_interp, sigma=3)
    centerline_smooth = np.column_stack((x_smooth, y_smooth)).astype(np.float32)

    # --- 1.1 Trim по процентам ---
    n = len(centerline_smooth)
    top_cut = int(n * trim_top_pct)
    bot_cut = int(n * trim_bottom_pct)
    if top_cut + bot_cut >= n - 2:
        raise ValueError("Слишком агрессивный trim%, центрлайн почти исчез")

    centerline_trimmed = centerline_smooth[top_cut : n - bot_cut]
    n = len(centerline_trimmed)

    # --- 1.2 Выпрямление концов ---
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

    # --- 2. Нормали ---
    dx = gaussian_filter1d(np.gradient(centerline_trimmed[:, 0]), sigma=3)
    dy = gaussian_filter1d(np.gradient(centerline_trimmed[:, 1]), sigma=3)
    lengths = np.hypot(dx, dy) + 1e-6
    normals = np.column_stack((-dy / lengths, dx / lengths))

    h_img, w_img = image.shape[:2]
    lines = []
    max_strip_width = 0

    num_points = len(centerline_trimmed)
    for i in range(num_points):
        cx, cy = centerline_trimmed[i]
        nx, ny = normals[i]

        length_neg, length_pos = 0, 0

        for step in range(1, max(h_img, w_img)):
            px, py = int(cx - nx * step), int(cy - ny * step)
            if not (0 <= px < w_img and 0 <= py < h_img) or mask[py, px] == 0:
                break
            length_neg += 1

        for step in range(1, max(h_img, w_img)):
            px, py = int(cx + nx * step), int(cy + ny * step)
            if not (0 <= px < w_img and 0 <= py < h_img) or mask[py, px] == 0:
                break
            length_pos += 1

        strip_width = length_neg + length_pos
        max_strip_width = max(max_strip_width, strip_width)

        line = []
        for j in range(-length_neg, length_pos):
            px = cx + nx * j
            py = cy + ny * j
            if 0 <= int(py) < h_img and 0 <= int(px) < w_img:
                coords_sample = np.array([[py], [px]], dtype=np.float32)
                pixel = np.stack(
                    [
                        map_coordinates(
                            image[:, :, c],
                            coords_sample,
                            order=1,
                            mode="reflect",
                        )[0]
                        for c in range(3)
                    ],
                    axis=-1,
                )
                line.append(pixel)
        lines.append(np.array(line, dtype=np.uint8))

    if not lines:
        raise ValueError("Не удалось построить развёртку: нет валидных линий")

    unwrapped = np.zeros((num_points, max_strip_width, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        if line.shape[0] < 2:
            continue
        resized_line = cv2.resize(
            line[None, :, :],
            (max_strip_width, 1),
            interpolation=cv2.INTER_LINEAR,
        )
        unwrapped[i] = resized_line[0]

    final = cv2.resize(unwrapped, (final_size, final_size), interpolation=cv2.INTER_LINEAR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


    return centerline_trimmed


def get_segmentation_mask(image, img_path):
    h_img, w_img = image.shape[:2]
    seg_results = seg_model(img_path)
    masks = seg_results[0].masks
    if masks is None:
        return None
    mask_tensor = masks.data[0].cpu().numpy()
    mask_resized = cv2.resize(mask_tensor, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    mask = (mask_resized * 255).astype(np.uint8)
    return mask

# ===============================
# ОБРАБОТКА ПАПОК
# ===============================

def process_folders(input_root, output_root):
    for folder_name in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        output_folder = os.path.join(output_root, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            input_path = os.path.join(folder_path, file_name)
            output_path = os.path.join(output_folder, file_name)

            if DEBUG:
                debug_path = os.path.join(
                    output_folder,
                    os.path.splitext(file_name)[0] + "_debug.jpg",
                )

            try:
                image = cv2.imread(input_path)
                if image is None:
                    print(f" Поврежденное изображение: {input_path}")
                    continue

                mask = get_segmentation_mask(image, input_path)
                if mask is None:
                    print(f" Маска не найдена: {input_path}")
                    continue

                centerline_raw = extract_smooth_centerline(mask)
                centerline_final = unwrap_belly_mask_only(
                    image=image,
                    mask=mask,
                    centerline_raw=centerline_raw,
                    save_path=output_path,
                )

                if DEBUG:
                    save_debug_centerlines(
                        image=image,
                        mask=mask,
                        centerline_raw=centerline_raw,
                        centerline_final=centerline_final,
                        save_path=debug_path,
                    )

                print(f" Успешно обработано: {input_path}")

            except Exception as e:
                print(f" Ошибка при обработке {input_path}: {e}")


if __name__ == "__main__":
    print(" Начало обработки...")
    process_folders(INPUT_ROOT, SEND_PHOTO_DIR)
    print(" Обработка завершена!")
    print(f" Все результаты находятся в: {SEND_PHOTO_DIR}")


'''

from ultralytics import YOLO


def train():
    # бери подходящий сегментационный предтрен, например yolo11s-seg
    model = YOLO("yolo11s-seg.pt")

    model.train(
        data="C:/klasss/archive/seg/data.yaml",
        task="segment",              # важно: это сегментация
        epochs=100,
        imgsz=640,

        # геометрия — аккуратная, чтобы не ломать центрлайн
        degrees=5.0,
        translate=0.05,
        scale=0.10,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        # цветовые аугментации
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,

        project="C:/klasss/archive/seg",
        name="yolo11s_seg_newts_exp1",
    )


if __name__ == "__main__":
    train()
'''
