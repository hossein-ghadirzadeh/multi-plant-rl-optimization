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
        import pygame
        pygame.init()
        self.pygame = pygame
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
            pygame.display.set_caption("Pong Multi-Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)
        else:
            self.screen = None
            self.clock = None
            self.font = None

        self.reset()

    def reset(self):
        # Paddles and ball
        self.left_y = (self.height - self.paddle_h)/2
        self.right_y = (self.height - self.paddle_h)/2
        self.ball_x = self.width/2
        self.ball_y = self.height/2
        import random, numpy as np
        angle = random.uniform(-0.25*np.pi,0.25*np.pi)
        direction = random.choice([-1,1])
        speed = random.uniform(3.0,4.5)
        self.ball_vel_x = direction*speed*np.cos(angle)
        self.ball_vel_y = speed*np.sin(angle)

        # Scores
        self.left_score = 0
        self.right_score = 0
        self.steps = 0
        self.done = False

        # ------------------ NEW METRICS ------------------
        self.rally_steps = 0
        self.left_rally_steps = 0
        self.right_rally_steps = 0

        # Reaction time tracking
        self.left_reaction_time = 0
        self.right_reaction_time = 0
        self.left_waiting_for_reaction = False
        self.right_waiting_for_reaction = False
        self.left_prev_y = self.left_y
        self.right_prev_y = self.right_y

        return self._get_state_left(), self._get_state_right()

    def _get_state_left(self):
        import numpy as np
        bx = self.ball_x/self.width
        by = self.ball_y/self.height
        vx = np.clip(self.ball_vel_x/6.0, -1.0, 1.0)
        vy = np.clip(self.ball_vel_y/6.0, -1.0, 1.0)
        py = self.left_y/(self.height-self.paddle_h)
        return np.array([bx,by,vx,vy,py], dtype=np.float32)

    def _get_state_right(self):
        import numpy as np
        bx = self.ball_x/self.width
        by = self.ball_y/self.height
        vx = np.clip(self.ball_vel_x/6.0, -1.0, 1.0)
        vy = np.clip(self.ball_vel_y/6.0, -1.0, 1.0)
        py = self.right_y/(self.height-self.paddle_h)
        return np.array([bx,by,vx,vy,py], dtype=np.float32)

    def step(self, left_action, right_action):
        import numpy as np
        # ------------------ Move paddles ------------------
        if left_action==0: self.left_y -= self.paddle_speed
        elif left_action==2: self.left_y += self.paddle_speed
        if right_action==0: self.right_y -= self.paddle_speed
        elif right_action==2: self.right_y += self.paddle_speed

        self.left_y = float(np.clip(self.left_y,0,self.height-self.paddle_h))
        self.right_y = float(np.clip(self.right_y,0,self.height-self.paddle_h))

        # ------------------ Move ball ------------------
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        # Bounce top/bottom
        if self.ball_y <= self.ball_radius:
            self.ball_y = self.ball_radius
            self.ball_vel_y *= -1
        elif self.ball_y >= self.height - self.ball_radius:
            self.ball_y = self.height - self.ball_radius
            self.ball_vel_y *= -1

        # Rewards
        reward_left = -0.001
        reward_right = -0.001
        info = {}
        rally_ended = False
        self.rally_steps += 1

        # ------------------ Collision / Miss ------------------
        # Left paddle
        if self.ball_x - self.ball_radius <= self.paddle_w:
            if self.left_y <= self.ball_y <= self.left_y+self.paddle_h:
                self.ball_x = self.paddle_w + self.ball_radius
                self.ball_vel_x = abs(self.ball_vel_x)+0.1
                offset = (self.ball_y-(self.left_y+self.paddle_h/2.0))/(self.paddle_h/2.0)
                self.ball_vel_y += offset*2.0
                reward_left += 0.05
                self.left_rally_steps += 1
            else:
                self.right_score += 1
                reward_right += 1.0
                rally_ended = True
                info['rally_winner']='right'

        # Right paddle
        if self.ball_x + self.ball_radius >= self.width - self.paddle_w:
            if self.right_y <= self.ball_y <= self.right_y+self.paddle_h:
                self.ball_x = self.width - self.paddle_w - self.ball_radius
                self.ball_vel_x = -abs(self.ball_vel_x)-0.1
                offset = (self.ball_y-(self.right_y+self.paddle_h/2.0))/(self.paddle_h/2.0)
                self.ball_vel_y += offset*2.0
                reward_right += 0.05
                self.right_rally_steps += 1
            else:
                self.left_score += 1
                reward_left += 1.0
                rally_ended = True
                info['rally_winner']='left'

        # ------------------ Reaction Time ------------------
        ball_toward_left = self.ball_vel_x < 0
        ball_toward_right = self.ball_vel_x > 0

        # Left agent reaction
        if ball_toward_left:
            if not self.left_waiting_for_reaction:
                self.left_waiting_for_reaction = True
                self.left_reaction_time = 0
            else:
                self.left_reaction_time += 1
            if self.left_y != self.left_prev_y and self.left_waiting_for_reaction:
                info['reaction_time_left'] = self.left_reaction_time
                self.left_waiting_for_reaction = False

        # Right agent reaction
        if ball_toward_right:
            if not self.right_waiting_for_reaction:
                self.right_waiting_for_reaction = True
                self.right_reaction_time = 0
            else:
                self.right_reaction_time += 1
            if self.right_y != self.right_prev_y and self.right_waiting_for_reaction:
                info['reaction_time_right'] = self.right_reaction_time
                self.right_waiting_for_reaction = False

        self.left_prev_y = self.left_y
        self.right_prev_y = self.right_y

        # ------------------ Rally Ended Metrics ------------------
        if rally_ended:
            info['rally_length_left'] = self.left_rally_steps
            info['rally_length_right'] = self.right_rally_steps
            self.rally_steps = 0
            self.left_rally_steps = 0
            self.right_rally_steps = 0

            # Reset ball for next rally if match not finished
            import random
            if self.left_score < self.max_points and self.right_score < self.max_points:
                self.ball_x = self.width/2
                self.ball_y = self.height/2
                angle = random.uniform(-0.25*np.pi,0.25*np.pi)
                direction = random.choice([-1,1])
                speed = random.uniform(3.0,4.5)
                self.ball_vel_x = direction*speed*np.cos(angle)
                self.ball_vel_y = speed*np.sin(angle)

        # ------------------ Check Done ------------------
        self.steps += 1
        done = False
        winner = None
        if self.left_score >= self.max_points:
            done = True
            winner='left'
        elif self.right_score >= self.max_points:
            done = True
            winner='right'
        elif self.steps >= MAX_TIMESTEPS:
            done = True
            winner='timeout'

        if done:
            self.done = True
            info['winner'] = winner
            info['final_score'] = (self.left_score, self.right_score)

        return self._get_state_left(), self._get_state_right(), float(reward_left), float(reward_right), done, info

    def render(self):
        if not self.render_mode: return
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit()
        self.screen.fill((0,0,0))
        pygame.draw.line(self.screen,(255,255,255),(self.width//2,0),(self.width//2,self.height),1)
        pygame.draw.rect(self.screen,(0, 0, 255),pygame.Rect(0,int(self.left_y),self.paddle_w,self.paddle_h))
        pygame.draw.rect(self.screen,(255, 0, 0),pygame.Rect(self.width-self.paddle_w,int(self.right_y),self.paddle_w,self.paddle_h))
        pygame.draw.circle(self.screen,(255, 255, 0),(int(self.ball_x),int(self.ball_y)),self.ball_radius)
        if self.font:
            left_label = self.font.render(f"Q-learning Score: {self.left_score}", True, (200,200,255))
            right_label = self.font.render(f"DQN Score: {self.right_score}", True, (200,200,255))
            self.screen.blit(left_label,(10,10))
            self.screen.blit(right_label,(self.width-150,10))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.render_mode:
            self.pygame.quit()
