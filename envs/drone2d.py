import numpy as np
import pygame
import gymnasium as gym
import Box2D


# pixels per meter
# world is 5x5 meters, windows is 512x512 pixels
PPM = 102.4
FPS = 30

MAX_FORCE = 2
MAX_SPEED = 100
MAX_ANGULAR_SPEED = 100
GRAVITY = -9.81

DRONE_DEF = Box2D.b2FixtureDef(density=4.0, friction=0.1, restitution=0.0, shape=Box2D.b2PolygonShape(box=(0.2, 0.05)))
GROUND_DEF = Box2D.b2FixtureDef(shape=Box2D.b2PolygonShape(box=(2.5, 0.1)))


class Drone2D(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        self.action_space = gym.spaces.Box(0, MAX_FORCE, (2, ), dtype=np.float32)
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._destroy()
        self.drone = self.world.CreateDynamicBody(position=(2.5, 0.25), fixtures=DRONE_DEF)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        reward = 10 - np.sqrt((self.drone.position[0] - 2.5 - 2) ** 2 + (self.drone.position[1] - 0.25 - 3) ** 2)

        self.drone.ApplyForce(force=self.drone.GetWorldVector([0., float(action[0])]), point=self.drone.GetWorldPoint([-0.2, 0.]), wake=True)
        self.drone.ApplyForce(force=self.drone.GetWorldVector([0., float(action[1])]), point=self.drone.GetWorldPoint([0.2, 0.]), wake=True)
        self.drone.ApplyForceToCenter(force=-1 * np.sign(self.drone.linearVelocity) * np.square(self.drone.linearVelocity), wake=True)
        self.world.Step(self.TIME_STEP, 10, 10)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, False, False, self._get_info()

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
        action = [1.6, 1.61]
    if keys[pygame.K_d]:
        action = [1.61, 1.6]
    return action


def main():
    env = Drone2D(render_mode="human")
    obs, info = env.reset(seed=0)

    for _ in range(1000):
        # action = [9.81*4*0.4*0.1, 9.81*4*0.4*0.1]
        keys = pygame.key.get_pressed()
        action = action_from_keyboard(keys)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated or keys[pygame.K_q]:
             obs, info = env.reset()

    env.close()


if __name__ == '__main__':
    main()
