import numpy as np

# =========================
# IOU FUNCTION
# =========================
def iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    inter = w * h
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    return inter / (area1 + area2 - inter + 1e-6)


# =========================
# TRACKER CLASS
# =========================
class Track:
    count = 0

    def __init__(self, bbox):
        self.id = Track.count
        Track.count += 1

        self.bbox = np.array(bbox, dtype=float)
        self.hits = 1
        self.age = 0

    def update(self, bbox):
        self.bbox = np.array(bbox, dtype=float)
        self.hits += 1
        self.age = 0

    def predict(self):
        self.age += 1
        return self.bbox


# =========================
# SORT TRACKER
# =========================
class Sort:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []

    def update(self, detections):
        results = []

        # No detections
        if len(detections) == 0:
            self._age_tracks()
            return np.empty((0, 5))

        detections = np.array(detections)

        used_dets = set()

        # Match detections to existing tracks
        for track in self.tracks:
            best_iou = 0
            best_det = -1

            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                score = iou(track.bbox, det)
                if score > best_iou:
                    best_iou = score
                    best_det = i

            if best_iou >= self.iou_threshold:
                track.update(detections[best_det])
                used_dets.add(best_det)
            else:
                track.predict()

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in used_dets:
                self.tracks.append(Track(det))

        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks if t.age <= self.max_age
        ]

        # Output confirmed tracks
        for track in self.tracks:
            if track.hits >= self.min_hits:
                results.append(
                    np.append(track.bbox.astype(int), track.id)
                )

        return np.array(results)

    def _age_tracks(self):
        for track in self.tracks:
            track.predict()
        self.tracks = [
            t for t in self.tracks if t.age <= self.max_age
        ]
