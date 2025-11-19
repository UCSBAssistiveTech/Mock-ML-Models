import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

class HiddenMarkovModel:
    def __init__(self, states, observations, transitions, emissions, initial_probs):
        self.states = states
        self.observations = observations
        self.transitions = transitions
        self.emissions = emissions
        self.initial_probs = initial_probs
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.obs_to_idx = {o: i for i, o in enumerate(observations)}

    def generate_sequence(self, n_steps):
        state_sequence = []
        obs_sequence = []

        current_state_idx = np.random.choice(len(self.states), p=self.initial_probs)

        for _ in range(n_steps):
            current_state = self.states[current_state_idx]
            state_sequence.append(current_state)

            obs_probs = self.emissions[current_state_idx]
            obs_idx = np.random.choice(len(self.observations), p=obs_probs)
            obs_sequence.append(self.observations[obs_idx])

            trans_probs = self.transitions[current_state_idx]
            current_state_idx = np.random.choice(len(self.states), p=trans_probs)

        return state_sequence, obs_sequence

    def viterbi(self, obs_sequence):
        n_states = len(self.states)
        n_obs = len(obs_sequence)

        viterbi_matrix = np.zeros((n_states, n_obs))
        backpointer = np.zeros((n_states, n_obs), dtype=int)

        obs_idx = self.obs_to_idx[obs_sequence[0]]
        viterbi_matrix[:, 0] = self.initial_probs * self.emissions[:, obs_idx]

        for t in range(1, n_obs):
            obs_idx = self.obs_to_idx[obs_sequence[t]]
            for s in range(n_states):
                trans_probs = viterbi_matrix[:, t-1] * self.transitions[:, s]
                backpointer[s, t] = np.argmax(trans_probs)
                viterbi_matrix[s, t] = np.max(trans_probs) * self.emissions[s, obs_idx]

        best_path = [0] * n_obs
        best_path[-1] = np.argmax(viterbi_matrix[:, -1])
        for t in range(n_obs-2, -1, -1):
            best_path[t] = backpointer[best_path[t+1], t+1]

        return [self.states[i] for i in best_path]


def visualize_hmm(hmm, state_seq, obs_seq):
    """Create visualization of HMM sequence"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    state_colors = {'Sunny': '#FFD700', 'Rainy': '#4682B4', 'Cloudy': '#A9A9A9'}
    obs_colors = {'Happy': '#90EE90', 'Grumpy': '#FF6B6B', 'Neutral': '#FFE4B5'}

    n_steps = len(state_seq)
    x_pos = range(n_steps)

    ax1.set_title('Hidden States (Weather)', fontsize=16, fontweight='bold')
    for i, state in enumerate(state_seq):
        ax1.add_patch(mpatches.Rectangle((i-0.4, 0), 0.8, 1,
                                         facecolor=state_colors.get(state, '#CCCCCC'),
                                         edgecolor='black', linewidth=2))
        ax1.text(i, 0.5, state, ha='center', va='center', fontsize=10, fontweight='bold')

    ax1.set_xlim(-0.5, n_steps-0.5)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x_pos)
    ax1.set_yticks([])
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.grid(True, axis='x', alpha=0.3)

    ax2.set_title('Observations (Mood)', fontsize=16, fontweight='bold')
    for i, obs in enumerate(obs_seq):
        ax2.add_patch(mpatches.Rectangle((i-0.4, 0), 0.8, 1,
                                         facecolor=obs_colors.get(obs, '#CCCCCC'),
                                         edgecolor='black', linewidth=2))
        ax2.text(i, 0.5, obs, ha='center', va='center', fontsize=10, fontweight='bold')

    ax2.set_xlim(-0.5, n_steps-0.5)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x_pos)
    ax2.set_yticks([])
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.grid(True, axis='x', alpha=0.3)

    ax3.set_title('HMM Structure', fontsize=16, fontweight='bold')
    ax3.text(0.5, 0.9, 'Transition Probabilities:', fontsize=12, fontweight='bold',
             transform=ax3.transAxes)
    ax3.text(0.5, 0.75, f'P(Sunny→Sunny)=0.8, P(Sunny→Rainy)=0.2',
             fontsize=10, transform=ax3.transAxes)
    ax3.text(0.5, 0.65, f'P(Rainy→Sunny)=0.4, P(Rainy→Rainy)=0.6',
             fontsize=10, transform=ax3.transAxes)

    ax3.text(0.5, 0.45, 'Emission Probabilities:', fontsize=12, fontweight='bold',
             transform=ax3.transAxes)
    ax3.text(0.5, 0.3, f'P(Happy|Sunny)=0.8, P(Grumpy|Sunny)=0.2',
             fontsize=10, transform=ax3.transAxes)
    ax3.text(0.5, 0.2, f'P(Happy|Rainy)=0.3, P(Grumpy|Rainy)=0.7',
             fontsize=10, transform=ax3.transAxes)

    ax3.axis('off')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    states = ['Sunny', 'Rainy']
    observations = ['Happy', 'Grumpy']

    transitions = np.array([
        [0.8, 0.2],  # From Sunny
        [0.4, 0.6]   # From Rainy
    ])

    emissions = np.array([
        [0.8, 0.2],  # Sunny -> Happy/Grumpy
        [0.3, 0.7]   # Rainy -> Happy/Grumpy
    ])

    initial_probs = np.array([0.6, 0.4])

    hmm = HiddenMarkovModel(states, observations, transitions, emissions, initial_probs)

    n_steps = 15
    state_seq, obs_seq = hmm.generate_sequence(n_steps)

    print("Generated Sequence:")
    print("-" * 50)
    for i, (state, obs) in enumerate(zip(state_seq, obs_seq)):
        print(f"Step {i+1}: State={state:8s} | Observation={obs}")
    print("-" * 50)

    predicted_states = hmm.viterbi(obs_seq)

    print("\nViterbi Prediction vs Actual:")
    print("-" * 50)
    accuracy = sum(p == a for p, a in zip(predicted_states, state_seq)) / len(state_seq)
    for i, (pred, actual) in enumerate(zip(predicted_states, state_seq)):
        match = "✓" if pred == actual else "✗"
        print(f"Step {i+1}: Predicted={pred:8s} | Actual={actual:8s} {match}")
    print("-" * 50)
    print(f"Accuracy: {accuracy*100:.1f}%")

    fig = visualize_hmm(hmm, state_seq, obs_seq)
    plt.show()
