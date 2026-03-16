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