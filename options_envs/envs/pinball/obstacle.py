from itertools import tee

import numpy as np


class PinballObstacle:
    def __init__(self, points):
        self.points = np.array(points, dtype=float)

        self.min_x = min(self.points, key=lambda pt: pt[0])[0]
        self.max_x = max(self.points, key=lambda pt: pt[0])[0]
        self.min_y = min(self.points, key=lambda pt: pt[1])[1]
        self.max_y = max(self.points, key=lambda pt: pt[1])[1]

        self._double_collision = False
        self._intercept = None

    def get_points(self):
        return self.points

    def collision(self, ball):
        self._double_collision = False

        if ball.position[0] - ball.radius > self.max_x:
            return False
        if ball.position[0] + ball.radius < self.min_x:
            return False
        if ball.position[1] - ball.radius > self.max_y:
            return False
        if ball.position[1] + ball.radius < self.min_y:
            return False

        a, b = tee(np.vstack([self.points, self.points[0]]))
        next(b, None)

        intercept_found = False

        for pt_pair in zip(a, b):
            if self._intercept_edge(pt_pair, ball):
                if intercept_found:
                    self._intercept = self._select_edge(pt_pair, self._intercept, ball)
                    self._double_collision = True
                else:
                    self._intercept = pt_pair
                    intercept_found = True

        return intercept_found

    def collision_effect(self, ball):
        if self._double_collision:
            return [-ball.xdot, -ball.ydot]

        obstacle_vector = self._intercept[1] - self._intercept[0]
        if obstacle_vector[0] < 0:
            obstacle_vector = self._intercept[0] - self._intercept[1]

        velocity_vector = np.array([ball.xdot, ball.ydot])

        theta = self._angle(velocity_vector, obstacle_vector) - np.pi
        if theta < 0:
            theta += 2 * np.pi

        intercept_theta = self._angle([-1, 0], obstacle_vector)
        theta += intercept_theta

        if theta > 2 * np.pi:
            theta -= 2 * np.pi

        speed = np.linalg.norm(velocity_vector)
        return [speed * np.cos(theta), speed * np.sin(theta)]

    def _select_edge(self, intersect1, intersect2, ball):
        velocity = np.array([ball.xdot, ball.ydot])

        obstacle_vector1 = intersect1[1] - intersect1[0]
        obstacle_vector2 = intersect2[1] - intersect2[0]

        angle1 = self._angle(velocity, obstacle_vector1)
        if angle1 > np.pi:
            angle1 -= np.pi

        angle2 = self._angle(velocity, obstacle_vector2)
        if angle2 > np.pi:
            angle2 -= np.pi

        if abs(angle1 - (np.pi / 2.0)) < abs(angle2 - (np.pi / 2.0)):
            return intersect1
        return intersect2

    def _intercept_edge(self, pt_pair, ball):
        p1, p2 = pt_pair

        obstacle_edge = p2 - p1
        difference = np.array(ball.position) - p1

        scalar_proj = difference.dot(obstacle_edge) / obstacle_edge.dot(obstacle_edge)
        if scalar_proj > 1.0:
            scalar_proj = 1.0
        elif scalar_proj < 0.0:
            scalar_proj = 0.0

        closest_pt = p1 + obstacle_edge * scalar_proj
        obstacle_to_ball = ball.position - closest_pt
        distance = obstacle_to_ball.dot(obstacle_to_ball)

        if distance <= ball.radius * ball.radius:
            velocity = np.array([ball.xdot, ball.ydot])
            ball_to_obstacle = closest_pt - ball.position

            angle = self._angle(ball_to_obstacle, velocity)
            if angle > np.pi:
                angle = 2 * np.pi - angle

            if angle > np.pi / 1.99:
                return False

            return True

        return False

    def _angle(self, v1, v2):
        angle_diff = np.arctan2(v1[0], v1[1]) - np.arctan2(v2[0], v2[1])
        if angle_diff < 0:
            angle_diff += 2 * np.pi
        return angle_diff
