import numpy as np

class GameLogic:
    def __init__(self):
        self.circles = []
        self.attached_circle = None

    def spawn_circle(self, box, specific_places=None):
        if specific_places:
            x, y = specific_places[np.random.randint(0, len(specific_places))]
        else:
            x = np.random.randint(box[0], box[2])
            y = np.random.randint(box[1], box[3])
        self.circles.append((x, y))

    def detect_collision(self, hand_pos, circle):
        dist = np.linalg.norm([hand_pos[0] - circle['x'], hand_pos[1] - circle['y']])
        return dist < circle['radius']

    def attach_circle(self, hand_pos):
        for circle in self.circles:
            if self.detect_collision(hand_pos, circle):
                self.attached_circle = circle
                break

    def move_attached_circle(self, hand_pos):
        if self.attached_circle:
            self.attached_circle['x'], self.attached_circle['y'] = hand_pos
