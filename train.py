from modeling import DQN
from transition import ReplayMemory, select_action
import matplotlib
import torch
import gym
import matplotlib.pyplot as plt
from input_extraction import get_screen
from itertools import count
import torch.optim as optim
import optimization

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def plot_durations(episode_durations=[]):
    """
    a helper for plotting the durations of episodes, along with an average over the last 100 episodes
    (the measure used in the official evaluations). The plot will be underneath the cell containing the
    main training loop, and will update after every episode.
    :return:
    """
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def run():
    # 创建环境并放开step限制（200）
    env = gym.make('CartPole-v0').unwrapped

    plt.ion()

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    # 环境初始化
    observation = env.reset()
    print(observation)

    plt.figure()
    plt.imshow(get_screen(env, device).cpu().squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()

    init_screen = get_screen(env, device)
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space(2)
    n_actions = env.action_space.n

    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)
    optimize_model = optimization.Optimizer(BATCH_SIZE, GAMMA, optimizer, memory, policy_net, target_net,device)

    steps_done = 0

    episode_durations = []
    num_episodes = 50

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(state, EPS_START, EPS_END,EPS_DECAY, device, policy_net, n_actions, steps_done)
            # 更新状态
            observation, reward, done, info = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            # print(observation)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env, device)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model.optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run()
