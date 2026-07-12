class BallModel:
    DRAG = 0.995

    def __init__(self, start_position, radius):
        self.position = list(start_position)
        self.radius = radius
        self.xdot = 0.0
        self.ydot = 0.0

    def add_impulse(self, delta_xdot, delta_ydot):
        self.xdot += delta_xdot / 5.0
        self.ydot += delta_ydot / 5.0
        self.xdot = self._clip(self.xdot)
        self.ydot = self._clip(self.ydot)

    def add_drag(self):
        self.xdot *= self.DRAG
        self.ydot *= self.DRAG

    def step(self):
        self.position[0] += self.xdot * self.radius / 20.0
        self.position[1] += self.ydot * self.radius / 20.0

    @staticmethod
    def _clip(val, low=-1, high=1):
        if val > high:
            val = high
        if val < low:
            val = low
        return val
