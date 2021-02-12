from collections import namedtuple
import random
import math
import torch


# a named tuple representing a single transition in our environment.
# It essentially maps (state, action) pairs to their (next_state, reward) result,
# with the state being the screen difference image as described later on
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """a cyclic buffer of bounded size that holds the transitions observed recently.
    It also implements a .sample() method for selecting a random batch of transitions for training.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state, EPS_START, EPS_END, EPS_DECAY, device, policy_net, n_actions, steps_done=0):
    """
     will select an action accordingly to an epsilon greedy policy. Simply put, we’ll sometimes use our model for
     choosing the action, and sometimes we’ll just sample one uniformly. The probability of choosing a random action
     will start at EPS_START and will decay exponentially towards EPS_END. EPS_DECAY controls the rate of the decay
    :param state:
    :return:
    """
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


