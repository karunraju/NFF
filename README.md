# Navigate, Fetch and Find
The goal is to build a reinforcement learning system that efficiently learns a high-reward policy.

## Environment
The world consists of a single agent which roams in an infinitely large grid world that
contains various objects (e.g., jelly beans, tongs, diamonds, immovable obstacles).
The agent can perform three possible actions: move forward, turn 90 degrees right, or turn 90
degrees left. The agents sensors include limited-range vision (of the nearby grid squares in the
direction the agent faces) and longer range smell (represented by a vector of different smell
dimensions). Different objects in the world have different visual appearance and different
smells. The agent receives an immediate reward of +20 for each jelly bean it eats (which
it achieves by stepping into any square that contains a jelly bean). The locations of jelly
beans on the grid is probabilistically related to locations of other objects. In addition, the
agent receives a reward of +100 if it picks up a diamond with tongs it is holding (if the agent
steps on a square containing a tong, it then automatically picks up and carries that tong
with it, and if it then steps into a square containing a diamond while carrying empty tongs,
it picks up that diamond and gets the reward, after which these tongs are no longer empty.
The agent may pick up and carry as many empty tongs as it likes)

## Training
```bash
$ python3 main.py
```

## Testing
```bash
$ python3 main.py --train 0 </path/to/model_file>
```

For rendering, add `--render 1`
