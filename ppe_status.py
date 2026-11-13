"""Shared per-person PPE evaluation and status audio."""

import os

from pygame import error as PygameError, mixer


PPE_PROPER = ['helmet', 'gloves', 'vest', 'boots', 'goggles']
PPE_VIOLATIONS = {
    'no_helmet': 'helmet',
    'no_goggle': 'goggles',
    'no_gloves': 'gloves',
    'no_boots': 'boots',
    'no_vest': 'vest',
    'none': None,
}
PERSON_CLASS = 'Person'
PASS_SOUND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pass.mp3')
FAIL_SOUND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fail.mp3')


def _overlap(person, item):
    x1, y1 = max(person[0], item[0]), max(person[1], item[1])
    x2, y2 = min(person[2], item[2]), min(person[3], item[3])
    area = (item[2] - item[0]) * (item[3] - item[1])
    if x2 <= x1 or y2 <= y1 or area <= 0:
        return 0.0
    return (x2 - x1) * (y2 - y1) / area


def check_violations(results, names, mode, required_ppe):
    detections = [
        {
            'class': names[int(box.cls[0])],
            'confidence': float(box.conf[0]),
            'bbox': box.xyxy[0].cpu().numpy(),
        }
        for box in results[0].boxes
    ]
    people = [d for d in detections if d['class'] == PERSON_CLASS]
    person_items = [[] for _ in people]
    for item in detections:
        if item['class'] not in PPE_PROPER and item['class'] not in PPE_VIOLATIONS:
            continue
        matches = [
            (
                _overlap(person['bbox'], item['bbox']),
                -((person['bbox'][0] + person['bbox'][2] - item['bbox'][0] - item['bbox'][2]) ** 2
                  + (person['bbox'][1] + person['bbox'][3] - item['bbox'][1] - item['bbox'][3]) ** 2),
                index,
            )
            for index, person in enumerate(people)
        ]
        if matches:
            overlap, _, index = max(matches)
            if overlap > 0.3:
                person_items[index].append(item['class'])

    evaluated = []
    for person, items in zip(people, person_items):
        ppe_status = {ppe: ppe in items for ppe in PPE_PROPER}
        violations = [item for item in items if item in PPE_VIOLATIONS]
        for violation in violations:
            missing_item = PPE_VIOLATIONS[violation]
            if missing_item is None:
                ppe_status = {ppe: False for ppe in PPE_PROPER}
            else:
                ppe_status[missing_item] = False

        relevant = violations if mode == 1 else [
            item for item in violations
            if PPE_VIOLATIONS[item] is None or PPE_VIOLATIONS[item] in required_ppe
        ]
        required = PPE_PROPER if mode == 1 else required_ppe
        missing = [item for item in required if not ppe_status.get(item, False)]
        if relevant:
            prefix = "Violation detected" if mode == 1 else "Required PPE violation"
            message = f"{prefix}: {', '.join(relevant)}"
        elif missing:
            prefix = "Missing PPE" if mode == 1 else "Missing required PPE"
            message = f"{prefix}: {', '.join(missing)}"
        else:
            message = None
        evaluated.append({
            'has_violation': message is not None,
            'violation_message': message,
            'ppe_status': ppe_status,
            'confidence': person['confidence'],
            'bbox': person['bbox'],
        })
    return evaluated


class StatusAudioMixin:
    def init_status_audio(self):
        self.last_audio_status = None
        try:
            mixer.init()
            self.audio_enabled = True
        except PygameError as exc:
            print(f"Warning: Could not initialize audio mixer - {exc}")
            self.audio_enabled = False
        for path in (PASS_SOUND_PATH, FAIL_SOUND_PATH):
            if not os.path.isfile(path):
                print(f"Warning: Status sound file not found at {path}")

    def play_status_sound(self, status):
        if status == self.last_audio_status:
            return
        self.last_audio_status = status
        if status is not None and self.audio_enabled:
            path = PASS_SOUND_PATH if status == "PASS" else FAIL_SOUND_PATH
            try:
                mixer.music.load(path)
                mixer.music.play()
            except (PygameError, OSError) as exc:
                print(f"Error playing {status} sound from {path}: {exc}")

    def stop_status_audio(self):
        self.last_audio_status = None
        if self.audio_enabled:
            mixer.music.stop()


class PersonTracker:
    def __init__(self, max_distance=50, memory_frames=30):
        self.max_distance = max_distance
        self.memory_frames = memory_frames
        self.reset()

    def update(self, detections):
        centroids = [(x + w // 2, y + h // 2) for x, y, w, h in detections]
        if not centroids:
            for person_id in list(self.tracked_persons):
                self.tracked_persons[person_id]['frames_missing'] += 1
                if self.tracked_persons[person_id]['frames_missing'] > self.memory_frames:
                    del self.tracked_persons[person_id]
            return []

        matched_ids = []
        used_ids = set()
        for centroid in centroids:
            candidates = [
                (
                    (centroid[0] - tracked['centroid'][0]) ** 2
                    + (centroid[1] - tracked['centroid'][1]) ** 2,
                    person_id,
                )
                for person_id, tracked in self.tracked_persons.items()
                if person_id not in used_ids
            ]
            candidates = [candidate for candidate in candidates if candidate[0] < self.max_distance ** 2]
            if candidates:
                _, person_id = min(candidates)
                self.tracked_persons[person_id]['centroid'] = centroid
                self.tracked_persons[person_id]['frames_missing'] = 0
                used_ids.add(person_id)
            else:
                person_id = self.next_id
                self.next_id += 1
                self.tracked_persons[person_id] = {
                    'centroid': centroid,
                    'frames_missing': 0,
                }
                used_ids.add(person_id)
            matched_ids.append(person_id)

        for person_id in list(self.tracked_persons):
            if person_id not in used_ids:
                self.tracked_persons[person_id]['frames_missing'] += 1
                if self.tracked_persons[person_id]['frames_missing'] > self.memory_frames:
                    del self.tracked_persons[person_id]
        return matched_ids

    def reset(self):
        self.tracked_persons = {}
        self.next_id = 1
