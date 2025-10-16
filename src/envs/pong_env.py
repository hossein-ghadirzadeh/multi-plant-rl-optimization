import sys
from pathlib import Path

import random
import numpy as np
import pygame

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import MAX_POINTS, WIDTH, HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED, BALL_RADIUS, FPS, MAX_TIMESTEPS

class PongEnv:
    def __init__(self, render_mode=False, max_points=MAX_POINTS):
        pygame.init()
        self.width = WIDTH
        self.height = HEIGHT
        self.paddle_w = PADDLE_WIDTH
        self.paddle_h = PADDLE_HEIGHT
        self.ball_radius = BALL_RADIUS
        self.paddle_speed = PADDLE_SPEED
        self.fps = FPS
        self.render_mode = render_mode
        self.max_points = max_points

        if self.render_mode:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pong Multi-Agent (Left: Q-learning, Right: DQN)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)
        else:
            self.screen = None
            self.clock = None
            self.font = None

        self.reset()

    def reset(self):
        # left paddle (Q-learning) and right paddle (DQN)
        self.left_y = (self.height - self.paddle_h) / 2.0
        self.right_y = (self.height - self.paddle_h) / 2.0

        self.ball_x = self.width / 2.0
        self.ball_y = self.height / 2.0

        angle = random.uniform(-0.25 * np.pi, 0.25 * np.pi)
        direction = random.choice([-1, 1])
        speed = random.uniform(3.0, 4.5)
        self.ball_vel_x = direction * speed * np.cos(angle)
        self.ball_vel_y = speed * np.sin(angle)

        # match scoring
        self.left_score = 0
        self.right_score = 0

        self.steps = 0
        self.done = False
        # return observations for both agents
        return self._get_state_left(), self._get_state_right()

    def _get_state_left(self):
        bx = self.ball_x / self.width
        by = self.ball_y / self.height
        vx = np.clip(self.ball_vel_x / 6.0, -1.0, 1.0)
        vy = np.clip(self.ball_vel_y / 6.0, -1.0, 1.0)
        py = self.left_y / (self.height - self.paddle_h)
        return np.array([bx, by, vx, vy, py], dtype=np.float32)

    def _get_state_right(self):
        # mirror frame so right agent sees normalized coordinates from its perspective
        bx = self.ball_x / self.width
        by = self.ball_y / self.height
        vx = np.clip(self.ball_vel_x / 6.0, -1.0, 1.0)
        vy = np.clip(self.ball_vel_y / 6.0, -1.0, 1.0)
        py = self.right_y / (self.height - self.paddle_h)
        # same features; agents can have symmetric state spaces
        return np.array([bx, by, vx, vy, py], dtype=np.float32)

    def step(self, left_action, right_action):
        # Actions: 0 up, 1 stay, 2 down
        if left_action == 0:
            self.left_y -= self.paddle_speed
        elif left_action == 2:
            self.left_y += self.paddle_speed
        if right_action == 0:
            self.right_y -= self.paddle_speed
        elif right_action == 2:
            self.right_y += self.paddle_speed

        # clamp paddles
        self.left_y = float(np.clip(self.left_y, 0, self.height - self.paddle_h))
        self.right_y = float(np.clip(self.right_y, 0, self.height - self.paddle_h))

        # Update ball
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        # Collide top/bottom
        if self.ball_y <= self.ball_radius:
            self.ball_y = self.ball_radius
            self.ball_vel_y *= -1
        elif self.ball_y >= self.height - self.ball_radius:
            self.ball_y = self.height - self.ball_radius
            self.ball_vel_y *= -1

        # Rewards for each agent this timestep (rally-level)
        reward_left = 0.0
        reward_right = 0.0
        info = {}
        rally_ended = False

        # Check left paddle collision / left miss (right scores)
        if self.ball_x - self.ball_radius <= self.paddle_w:
            if (self.left_y <= self.ball_y <= self.left_y + self.paddle_h):
                # bounce
                self.ball_x = self.paddle_w + self.ball_radius
                self.ball_vel_x = abs(self.ball_vel_x) + 0.1
                offset = (self.ball_y - (self.left_y + self.paddle_h / 2.0)) / (self.paddle_h / 2.0)
                self.ball_vel_y += offset * 2.0
                reward_left += 0.05  # small reward for hitting
            else:
                # right scores a point
                self.right_score += 1
                reward_right += 1.0
                rally_ended = True
                info['rally_winner'] = 'right'

        # Check right paddle collision / right miss (left scores)
        if self.ball_x + self.ball_radius >= self.width - self.paddle_w:
            if (self.right_y <= self.ball_y <= self.right_y + self.paddle_h):
                # bounce
                self.ball_x = self.width - self.paddle_w - self.ball_radius
                self.ball_vel_x = -abs(self.ball_vel_x) - 0.1
                offset = (self.ball_y - (self.right_y + self.paddle_h / 2.0)) / (self.paddle_h / 2.0)
                self.ball_vel_y += offset * 2.0
                reward_right += 0.05
            else:
                # left scores a point
                self.left_score += 1
                reward_left += 1.0
                rally_ended = True
                info['rally_winner'] = 'left'

        # small time penalty to encourage scoring
        reward_left -= 0.001
        reward_right -= 0.001

        # Cap speeds
        speed = np.sqrt(self.ball_vel_x ** 2 + self.ball_vel_y ** 2)
        max_speed = 8.0
        if speed > max_speed:
            scale = max_speed / speed
            self.ball_vel_x *= scale
            self.ball_vel_y *= scale

        # If a rally ended but nobody reached match max points -> reset rally (not done)
        done = False
        winner = None
        if rally_ended:
            # check match end
            if self.left_score >= self.max_points:
                done = True
                winner = 'left'
            elif self.right_score >= self.max_points:
                done = True
                winner = 'right'
            else:
                # reset ball for next rally in center with random direction
                self.ball_x = self.width / 2.0
                self.ball_y = self.height / 2.0
                angle = random.uniform(-0.25 * np.pi, 0.25 * np.pi)
                direction = random.choice([-1, 1])
                speed = random.uniform(3.0, 4.5)
                self.ball_vel_x = direction * speed * np.cos(angle)
                self.ball_vel_y = speed * np.sin(angle)
                # keep paddles where they are (or optionally reset) and continue
        self.steps += 1
        if self.steps >= MAX_TIMESTEPS:
            done = True
            winner = 'timeout'

        if done:
            self.done = True
            info['winner'] = winner
            info['final_score'] = (self.left_score, self.right_score)

        # return states for both agents
        return (self._get_state_left(), self._get_state_right(),
                float(reward_left), float(reward_right),
                done, info)

    def render(self):
        if not self.render_mode or not pygame.get_init():
            return
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
            self.screen.fill((0, 0, 0))
            # middle line
            pygame.draw.line(self.screen, (255, 255, 255), (self.width//2, 0), (self.width//2, self.height), 1)
            # paddles
            pygame.draw.rect(self.screen, (255,255,255), pygame.Rect(0, int(self.left_y), self.paddle_w, self.paddle_h))
            pygame.draw.rect(self.screen, (255,255,255), pygame.Rect(self.width - self.paddle_w, int(self.right_y), self.paddle_w, self.paddle_h))
            # ball
            pygame.draw.circle(self.screen, (255,255,255), (int(self.ball_x), int(self.ball_y)), self.ball_radius)

            # Draw scores and labels
            left_label = self.font.render(f"Left: Q-learning ({self.left_score})", True, (200,200,255))
            right_label = self.font.render(f"Right: DQN ({self.right_score})", True, (200,200,255))
            self.screen.blit(left_label, (10, 10))
            self.screen.blit(right_label, (self.width - 180, 10))

            pygame.display.flip()
            self.clock.tick(self.fps)
        except pygame.error:
            # Handle case when video system was closed
            print("[Warning] Render skipped — pygame window not active.")
            return


    def close(self):
        if self.render_mode:
            pygame.quit()
