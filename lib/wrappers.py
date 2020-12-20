import cv2
import gym
import gym.spaces
import numpy as np
import collections


# A wrapper that presses the FIRE button in envs that require it and also check corner cases that are present in some atari games
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        
        # get_action_meanings returns the meanings of available actions such as ['Left' 'Right' etc]. In few atari envs 'FIRE' is an option 
        # Checking for envs which require FIRE option
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    # Perform a step for the given action
    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        (obs, _, done, _) = self.env.step(1)
        if done:
            self.env.reset()
        (obs, _, done, _) = self.env.step(2)
        if done:
            self.env.reset()
        return obs


# Wrapper that takes the maximum of pixel values from the 'skip'# of frames.
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._skip = skip
        self._obs_buffer = collections.deque(maxlen=self._skip)

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            (obs, reward, done, info) = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            # Chosen action is simply repeated to increase training speed and also there will be minor difference
            if done:
                break
        # Now we stack the obs into a 4th dimension (as images are 3d) and then get the max for each value, resulting in same shape
        # ex. 4 * (120, 100, 3) -> (4, 120, 100, 3) -> (120, 100, 3)
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return (max_frame, total_reward, done, info)

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


# Wrapper that converts input observation from emulator to grayscale 84*84
class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32) # convert from int to float type
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."

        # Convert the image into grayscale using the formula
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


# This wrapper changes the image format from (height, width, channel) to (channel, width, height) opposite to tensorflow's
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

# Wrapper to convert observation data (0, 255) to every pixel value to the range [0.0, ..., 1.0], i.e scaling and float conversion
class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


# A Wrapper that creates a stack of subsequent frames along the 1st dimension and returns them as an observation. 
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0), # vertically stack the same array (repeating along axis 0)
            old_space.high.repeat(n_steps, axis=0), dtype=dtype)
    
    # We need to use this method before working with the other method, otherwise self.buffer wouldn't be initialized
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

# A simple function that takes the environments and applies all the required wrappers to it
def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    # Note that the wrappers change the environment observation value from a single image to resized, gray images (4 as per default) with some modifications, this helps in understanding dynamics and few more
    return ScaledFloatFrame(env)
