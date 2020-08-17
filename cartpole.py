import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v0')

# Cosntants
BIN_NUMBER = 10  # Number of discretized states
GAMMA = 0.9
ALPHA = 0.01
EPSILON = 0.1
MAX_STATES = BIN_NUMBER**4  # In cart pool problem, a state is defined by 4 params, thus powering with 4

def create_bins(env, bin_number):
    '''
    In cartpole problem, states are defined with position & velocity of the cart and the angle & tip
    velocity of the pole. Max and min values for the env variables can be accessed with env.observation_space
    :param bin_number: Number of discretized bins
    :return:
    '''

    bins = np.zeros((4, bin_number), dtype=np.float)
    bound_low = env.observation_space.low
    bound_high = env.observation_space.high
    for i in range(4):
        bins[i] = np.linspace(bound_low[i], bound_high[i], bin_number)
    return  bins

def assign_bins(observation, bins):
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
    return state

def initialize_Q(env, num_bins=10):
    num_actions = env.action_space.n
    Q = np.zeros((np.power(num_bins, 4), num_actions), dtype=np.int)
    return Q

def set_state(Q, state, value):
    curr_val = Q
    for index in state:
        curr_val = curr_val[index]
    curr_val = value

def get_state(Q, state):
    curr_val = Q
    for index in state:
        curr_val = curr_val[index]
    return curr_val

def encode_state(state):
    '''
    ONLY FOR DISCRETE STATES
    Discrete states are represented as d elemented arrays. In order to construct a Q table, these vectors must be
    transformed to a single integer. This can be achieved by considering the elements of the state as the terms
    in d-based number.
    :param state:
    :return:
    '''
    # First state is the most significant
    num = 0
    for i in range(4):
        num+= np.power(4,4-i)*state[i]
    return num

def decode_state(encoded_state, var_count):
    '''
    ONLY FOR DISCRETE STATES
    Decodes an encoded discrete state by transforming it to a d-based number. Encoded states are indices in the Q table
    :param encoded_state (int): An integer number representing the encoded state
    :param var_count (int): Number of variables that represents a state
    :return (array): Integer array of the state
    '''
    states = []
    for i in range(var_count):
        div = np.floor(encoded_state/np.power(var_count, var_count-i))
        rem = encoded_state - div*np.power(var_count, var_count-i)
        states.append(div)
        encoded_state = rem

    return np.array(states, dtype=np.int)

def play_game(env, bins, eps = 0.1):
    observation = env.reset()
    done = False
    count = 0
    Q = initialize_Q()
    state = get_state(Q, bins)
    encoded_state = encode_state(state)
    total_reward = 0

    while not done:
        count += 1
        if np.random.uniform() < eps:
            act = env.action_space.sample()  # Make an exploratory move
        else:
            actions = np.max(Q[encoded_state])[0]  # Get the index of max action
# A 4 elemented vector for the cartpole problem, representing the states (pos, vel, ang_pole, vel_pole
bins = create_bins(env, BIN_NUMBER)
Q = initialize_Q()
observation = env.reset()
states = assign_bins(observation, bins)
bestLength = 0
episode_lengths = []
best_weights = np.zeros(4)

for i in range(100):
    new_weighs = np.random.uniform(-1.0, 1.0, 4)
    length = []
    for j in range(100):
        observation = env.reset()
        done = False
        count = 0
        while not done:
            #env.render()
            count += 1
            action = 1 if np.dot(observation, new_weighs) > 0 else 0
            observation, reward, done, _ = env.step(action)

            if done:
                break
        length.append(count)
    average_length = float(sum(length)/len(length))
    if average_length > bestLength:
        bestLength = average_length
        best_weights = new_weighs

    episode_lengths.append(average_length)

done = False
count = 0
env = wrappers.Monitor(env, 'records', force=True)
observation = env.reset()
while not done:
    env.render()
    count += 1
    action = 1 if np.dot(observation, best_weights) > 0 else 0
    observation, reward, done, _ = env.step(action)

    if done:
        break
print('game lasted ', count, ' moves')