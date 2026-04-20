from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path, default_confidence=0.40):
        self.model = YOLO(model_path)
        self.class_names = self.model.names if hasattr(self.model, "names") else {0: "fire"}
        self.default_confidence = default_confidence

    def detect(self, frame_bgr, width, height, conf_thres=None):
        threshold = self.default_confidence if conf_thres is None else conf_thres
        results = self.model.predict(source=frame_bgr, conf=threshold, verbose=False)

        detections = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes_xyxy, confs, clss):
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            cx_box = int((x1 + x2) / 2)
            cy_box = int((y1 + y2) / 2)

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": self.class_names.get(cls_id, str(cls_id)),
                    "conf": float(conf),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx_box,
                    "cy": cy_box,
                }
            )

        return detections

