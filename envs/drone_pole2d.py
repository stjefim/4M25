import numpy as np
import pygame
import gymnasium as gym
import Box2D
import random

from reward_funcs import _calc_distance, InverseDistance

# pixels per meter
WORLD_SIZE = 5.  # world size is 5 meters
PPM = 512 / WORLD_SIZE
FPS = 30

MAX_FORCE = 8
MAX_TORQUE = 0.3  # scaled by arm length?
MAX_SPEED = 10
MAX_ANGULAR_SPEED = 5
GRAVITY = -9.81

DRONE_DENSITY = 23.
DRONE_FRICTION = 0.1
DRONE_RESTITUTION = 0.0
DRONE_WH = np.array([0.35, 0.075])
# category bits and mask bits are used to make drone and pole collide with ground and not collide between each other
DRONE_DEF = Box2D.b2FixtureDef(density=DRONE_DENSITY, friction=DRONE_FRICTION, restitution=DRONE_RESTITUTION,
                               shape=Box2D.b2PolygonShape(box=DRONE_WH / 2), categoryBits=0x0001, maskBits=0x0002)

GROUND_DEF = Box2D.b2FixtureDef(shape=Box2D.b2PolygonShape(box=(WORLD_SIZE / 2, 0.1)), categoryBits=0x0002, maskBits=0x0001)

POLE_DENSITY = 4.0
POLE_FRICTION = 0.1
POLE_RESTITUTION = 0.0
POLE_WH = np.array([0.05, 0.5])
POLE_DEF = Box2D.b2FixtureDef(density=POLE_DENSITY, friction=POLE_FRICTION, restitution=POLE_RESTITUTION,
                              shape=Box2D.b2PolygonShape(box=(POLE_WH[0] / 2, POLE_WH[1] / 2, (0, POLE_WH[1] / 2), 0)), categoryBits=0x0001, maskBits=0x0002)

TARGET_MAX_X = 2
TARGET_MAX_Y = 4
TARGET_DISTANCE_THRESH = 0.5

LINEAR_DRAG_K = 0.5
ROTATIONAL_DRAG_K = 0.02


class DronePole2D(gym.Env):
    # Define some metadata about the environment, including the available render modes and the FPS.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    # Define some constants for the action types that this environment supports.
    ACTION_FORCES = 0
    ACTION_FORCE_AND_TORQUE = 1

    def __init__(self, reward_func, multiple_obj=False, render_mode=None, action_type=ACTION_FORCES,
                 spawn_position=(0, 3), initial_target_pos=None):
        # Check that the specified action type is valid.
        assert action_type == self.ACTION_FORCES or action_type == self.ACTION_FORCE_AND_TORQUE
        self.action_type = action_type

        # Define the action space for the environment based on the selected action type.
        if self.action_type == self.ACTION_FORCES:
            self.action_space = gym.spaces.Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        elif self.action_type == self.ACTION_FORCE_AND_TORQUE:
            self.action_space = gym.spaces.Box(np.array([0.0, -1.0]), np.array([1.0, 1.0]))

        # Define the observation space for the environment based on the maximum values of each observation variable.
        dims_min = np.array(
            [-WORLD_SIZE / 2, -0.25, -np.pi, -np.pi, -MAX_SPEED, -MAX_SPEED, -MAX_ANGULAR_SPEED, -MAX_ANGULAR_SPEED,
             -TARGET_MAX_X, 0])
        dims_max = np.array(
            [WORLD_SIZE / 2, 4.75, np.pi, np.pi, MAX_SPEED, MAX_SPEED, MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED,
             TARGET_MAX_X, TARGET_MAX_Y])

        self.observation_space = gym.spaces.Box(dims_min, dims_max)

        # Set the size of the rendering window and calculate the time step based on the render FPS.
        self.window_size = 512
        self.TIME_STEP = 1.0 / self.metadata["render_fps"]

        # Check that the specified render mode is valid.
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialise the rendering window and clock
        self.window = None
        self.clock = None

        # Create the Box2D world, ground body, drone body, and target position.
        self.world = Box2D.b2World(gravity=(0, GRAVITY))
        self.world.CreateStaticBody(position=(0, -0.15), fixtures=GROUND_DEF)
        self.spawn_position = spawn_position
        self.drone = self.world.CreateDynamicBody(position=self.spawn_position, fixtures=DRONE_DEF)
        self.pole = self.world.CreateDynamicBody(position=self.spawn_position, fixtures=POLE_DEF)
        self.joint = self.world.CreateRevoluteJoint(bodyA=self.drone, bodyB=self.pole, anchor=self.spawn_position)

        if initial_target_pos is not None:
            self.target = np.array(initial_target_pos)
        else:
            self.target = np.zeros(2)
            self._generate_target_position()

        # Initialise the action to None, this is needed to render the green force indicators
        self.action = None

        self.reward_func = reward_func
        self.multiple_obj = multiple_obj

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Destroy the previous drone instance
        self._destroy()
        # Create a new drone instance with a fixed position and fixtures defined by the DRONE_DEF constant
        self.drone = self.world.CreateDynamicBody(position=self.spawn_position, fixtures=DRONE_DEF)
        self.pole = self.world.CreateDynamicBody(position=self.spawn_position, fixtures=POLE_DEF)
        self.joint = self.world.CreateRevoluteJoint(bodyA=self.drone, bodyB=self.pole, anchor=self.spawn_position)

        self.pole.ApplyForceToCenter(force=[random.uniform(-0.1, 0.1), 0], wake=True)

        # If the render mode is set to "human", render a frame
        if self.render_mode == "human":
            self._render_frame()

        # Returns the observation and info dictionaries
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.action = self._scale_actions(action)

        # Apply engine forces
        self.drone.ApplyForce(force=self.drone.GetWorldVector([0., float(self.action[0])]),
                              point=self.drone.GetWorldPoint([-0.2, 0.]), wake=True)
        self.drone.ApplyForce(force=self.drone.GetWorldVector([0., float(self.action[1])]),
                              point=self.drone.GetWorldPoint([0.2, 0.]), wake=True)
        # Apply drag
        self.drone.ApplyForceToCenter(
            force=-LINEAR_DRAG_K * np.sign(self.drone.linearVelocity) * np.square(self.drone.linearVelocity), wake=True)
        self.drone.ApplyTorque(-ROTATIONAL_DRAG_K * np.sign(self.drone.angularVelocity) * np.square(self.drone.angularVelocity), wake=True)

        # Step the world forward in time
        self.world.Step(self.TIME_STEP, 10, 10)

        reward, terminated = self.reward_func(self._get_obs(), self.target)

        if self.multiple_obj:
            distance = _calc_distance(self._get_obs(), self.target)
            if distance < TARGET_DISTANCE_THRESH:
                self._generate_target_position()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # Initialize Pygame display if render mode is "human" and window is not already initialized
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        # Create a Pygame clock if render mode is "human" and clock is not already initialized
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create a Pygame surface and fill it with white
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # background color - white

        # Draw all bodies in the world
        for body in self.world.bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                vertices = np.array([body.transform * v for v in shape.vertices])
                pygame.draw.polygon(canvas, (100, 100, 100), self._coord_transform(vertices))

        # Draw target circle and forces on canvas
        pygame.draw.circle(canvas, (0, 200, 0), np.round(self._coord_transform(self.target)).astype(int), 10)
        self._draw_force(canvas, [-0.195, 0], 0)
        self._draw_force(canvas, [0.185, 0], 1)

        # If render mode is "human", display the canvas on the Pygame window
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        # If render mode is "rgb_array", return the rendered frame
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        # Quit Pygame display and Pygame modules
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        # Return observation vector containing drone position, angle, linear and angular velocities
        return np.array([self.drone.position[0], self.drone.position[1], self.drone.angle, self.joint.angle,
                         self.drone.linearVelocity[0], self.drone.linearVelocity[1], self.drone.angularVelocity,
                         self.joint.speed, self.target[0], self.target[1]]).astype(np.float32)

    def _get_info(self):
        # Return empty dictionary as the "info" dictionary is not used in this environment
        return {}

    def _coord_transform(self, coords_physics):
        # Convert physics coordinates to pixel coordinates using pixels per meter (PPM) conversion factor
        coords_physics = np.array(coords_physics)
        coords_physics = coords_physics + [2.5, 0.25]
        result = coords_physics * PPM

        # If result is a single point, transform its y-coordinate to account for Pygame coordinate system
        if result.shape == (2,):
            result[1] = self.window_size - result[1]
            return result

        # If result is a set of points, transform each point's y-coordinate to account for Pygame coordinate system
        result[:, 1] = self.window_size - result[:, 1]
        return result

    def _destroy(self):
        self.world.DestroyJoint(self.joint)
        self.world.DestroyBody(self.drone)
        self.world.DestroyBody(self.pole)

    def _draw_force(self, canvas, local_pos, action_index):
        # Draw line representing force being applied to drone
        local_pos = np.array(local_pos)

        # Get starting point of line by transforming local_pos to world coordinates and then to pixel coordinates
        start = np.round(self._coord_transform(self.drone.GetWorldPoint(local_pos))).astype(int)

        # Get ending point of line by applying action to local_pos, transforming to world coordinates, and then to pixel coordinates
        end = np.round(self._coord_transform(self.drone.GetWorldPoint(local_pos + [0., 0.2]))).astype(int)

        # Draw line representing direction of force
        pygame.draw.line(canvas, (200, 200, 200), start, end, width=3)

        # If an action has been taken, draw line representing resulting force
        if self.action is not None:
            endf = np.round(self._coord_transform(self.drone.GetWorldPoint(
                local_pos + np.array([0., 0.2]) * self.action[action_index] / MAX_FORCE))).astype(int)
            pygame.draw.line(canvas, (0, 200, 0), start, endf, width=3)

    def _generate_target_position(self):
        self.target[0] = random.uniform(-TARGET_MAX_X, TARGET_MAX_X)
        self.target[1] = random.uniform(0, TARGET_MAX_Y)

    def _scale_actions(self, action):
        action = np.array(action)
        # Scale the action if the action type is forces
        if self.action_type == self.ACTION_FORCES:
            action *= MAX_FORCE
        # Scale and convert to 2 force representation
        if self.action_type == self.ACTION_FORCE_AND_TORQUE:
            total_force = action[0] * MAX_FORCE * 2
            torque = action[1] * MAX_TORQUE
            action[0] = (total_force + torque) / 2
            action[1] = (total_force - torque) / 2
        return action


def action_from_keyboard(keys):
    action = [0, 0]
    if keys[pygame.K_w]:
        action = [5.0, 5.0]
    if keys[pygame.K_a]:
        action = [5.0, 5.1]
    if keys[pygame.K_d]:
        action = [5.1, 5.0]
    return np.array(action) / MAX_FORCE


def action_from_keyboard_gravity_cancel(keys):
    action = [-GRAVITY * np.prod(DRONE_WH) * DRONE_DENSITY, 0]
    if keys[pygame.K_w]:
        action = [-GRAVITY * 0.1594 + 0.1, 0]
    if keys[pygame.K_s]:
        action = [-GRAVITY * 0.1594 - 0.1, 0]
    if keys[pygame.K_a]:
        action = [-GRAVITY * 0.1594, -0.2]
    if keys[pygame.K_d]:
        action = [-GRAVITY * 0.1594, 0.2]

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
        env = DronePole2D(render_mode="human", action_type=DronePole2D.ACTION_FORCE_AND_TORQUE, reward_func=InverseDistance(),
                      multiple_obj=True)
        joystick = Joystick()
    else:
        env = DronePole2D(render_mode="human", action_type=DronePole2D.ACTION_FORCES, reward_func=InverseDistance(),
                      multiple_obj=True)

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
