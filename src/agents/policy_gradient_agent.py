import torch
import torch.nn as nn
import torch.optim as optim

# policy network for choosing actions
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, num_plants, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_size, hidden), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, 3)
                                   for _ in range(num_plants)])

    def forward(self, x):
        x = self.shared(x)
        return [h(x) for h in self.heads]

# policy gradient agent
class PolicyGradientAgent:
    def __init__(self, env, lr=1e-3, gamma=0.95):
        self.env = env
        self.gamma = gamma
        self.model = PolicyNetwork(
            state_size=env.num_plants + 1, num_plants=env.num_plants)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.log_probs, self.rewards = [], []

    def choose_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0)  # convert state to tensor
        logits = self.model(s)
        acts, logs = [], []
        # sample actions for each plant
        for logit in logits:
            p = torch.distributions.Categorical(logits=logit)
            a = p.sample()
            acts.append(a.item())
            logs.append(p.log_prob(a))
               # store log probabilities
        self.log_probs.append(torch.stack(logs).sum())
        return acts

    def store_reward(self, r):
        self.rewards.append(r)

    def finish_episode(self):
        # compute discounted rewards
        R, returns = 0, []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
         # normalize rewards
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # compute policy gradient loss
        loss = sum(-lp * G for lp, G in zip(self.log_probs, returns))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # clear stored log probabilities and rewards
        self.log_probs.clear()
        self.rewards.clear()
