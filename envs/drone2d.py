import numpy as np
import pygame
import gymnasium as gym
import Box2D


# pixels per meter
# world is 5x5 meters, window is 512x512 pixels
PPM = 102.4
FPS = 30

MAX_FORCE = 3
MAX_TORQUE = 0.05  # scaled by arm length?
MAX_SPEED = 100
MAX_ANGULAR_SPEED = 100
GRAVITY = -9.81

DRONE_DEF = Box2D.b2FixtureDef(density=4.0, friction=0.1, restitution=0.0, shape=Box2D.b2PolygonShape(box=(0.2, 0.05)))
GROUND_DEF = Box2D.b2FixtureDef(shape=Box2D.b2PolygonShape(box=(2.5, 0.1)))


class Drone2D(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    ACTION_FORCES = 0
    ACTION_FORCE_AND_TORQUE = 1

    def __init__(self, render_mode=None, action_type=ACTION_FORCES):
        assert action_type == self.ACTION_FORCES or action_type == self.ACTION_FORCE_AND_TORQUE
        self.action_type = action_type
        if self.action_type == self.ACTION_FORCES:
            self.action_space = gym.spaces.Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]), dtype=np.float32)
        elif self.action_type == self.ACTION_FORCE_AND_TORQUE:
            self.action_space = gym.spaces.Box(np.array([0.0, -1.0]), np.array([1.0, 1.0]), dtype=np.float32)

        dims = np.array([2.5, 2.5, np.pi, np.pi, MAX_SPEED, MAX_SPEED, MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED]).astype(np.float32)
        self.observation_space = gym.spaces.Box(-dims, dims)

        self.window_size = 512
        self.TIME_STEP = 1.0 / self.metadata["render_fps"]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.world = Box2D.b2World(gravity=(0, GRAVITY))
        self.world.CreateStaticBody(position=(2.5, 0.1), fixtures=GROUND_DEF)
        self.drone = self.world.CreateDynamicBody(position=(2.5, 0.25), fixtures=DRONE_DEF)
        self.target = np.array([1., 3.])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._destroy()
        self.drone = self.world.CreateDynamicBody(position=(2.5, 0.25), fixtures=DRONE_DEF)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        terminated = False
        reward = -100*((self.drone.position[0] - 2.5 - self.target[0]) ** 2 + (self.drone.position[1] - 0.25 - self.target[1]) ** 2)

        action = np.array(action)
        if self.action_type == self.ACTION_FORCES:
            action *= MAX_FORCE
        if self.action_type == self.ACTION_FORCE_AND_TORQUE:
            total_force = action[0] * MAX_FORCE * 2
            torque = action[1] * MAX_TORQUE
            action[0] = (total_force + torque) / 2
            action[1] = (total_force - torque) / 2

        self.drone.ApplyForce(force=self.drone.GetWorldVector([0., float(action[0])]), point=self.drone.GetWorldPoint([-0.2, 0.]), wake=True)
        self.drone.ApplyForce(force=self.drone.GetWorldVector([0., float(action[1])]), point=self.drone.GetWorldPoint([0.2, 0.]), wake=True)
        self.drone.ApplyForceToCenter(force=-0.5 * np.sign(self.drone.linearVelocity) * np.square(self.drone.linearVelocity), wake=True)
        self.drone.ApplyTorque(-0.01 * self.drone.angularVelocity, wake=True)
        self.world.Step(self.TIME_STEP, 10, 10)

        if self.drone.position[0] > 5 or self.drone.position[0] < 0 or self.drone.position[1] > 5 or \
                self.drone.angle > np.pi / 2 or self.drone.angle < -np.pi / 2:
            reward = -10000
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        for body in self.world.bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                vertices = np.array([body.transform * v for v in shape.vertices])
                pygame.draw.polygon(canvas, (100, 100, 100), self._coord_transform(vertices))
        self.world.Step(self.TIME_STEP, 10, 10)

        pygame.draw.circle(canvas, (0, 200, 0), np.round(self._coord_transform(self.target + [2.5, 0.25])).astype(int), 10)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        return np.array([self.drone.position[0] - 2.5, self.drone.position[1] - 0.25, self.drone.angle, 0,
                         self.drone.linearVelocity[0], self.drone.linearVelocity[1], self.drone.angularVelocity, 0]).astype(np.float32)

    def _get_info(self):
        return {}

    def _coord_transform(self, coords_physics):
        result = coords_physics * PPM
        if result.shape == (2, ):
            result[1] = self.window_size - result[1]
            return result
        result[:, 1] = self.window_size - result[:, 1]
        return result

    def _destroy(self):
        self.world.DestroyBody(self.drone)


def action_from_keyboard(keys):
    action = [0, 0]
    if keys[pygame.K_w]:
        action = [1.6, 1.6]
    if keys[pygame.K_a]:
        action = [1.6, 1.62]
    if keys[pygame.K_d]:
        action = [1.62, 1.6]
    return np.array(action) / MAX_FORCE


class Joystick:
    def __init__(self):
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def get_action(self):
        total = (self.joystick.get_axis(2) + 1) / 2
        diff = self.joystick.get_axis(0)
        return np.array([total, diff])


def main():
    JOYSTICK = False
    KEYBOARD = True

    if JOYSTICK:
        env = Drone2D(render_mode="human", action_type=Drone2D.ACTION_FORCE_AND_TORQUE)
        joystick = Joystick()
    else:
        env = Drone2D(render_mode="human", action_type=Drone2D.ACTION_FORCES)

    obs, info = env.reset(seed=0)

    for _ in range(1000):
        action = [0, 0]
        keys = pygame.key.get_pressed()
        if KEYBOARD:
            action = action_from_keyboard(keys)
        if JOYSTICK:
            action = joystick.get_action()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated or keys[pygame.K_q]:
            obs, info = env.reset()

    env.close()


if __name__ == '__main__':
    main()
