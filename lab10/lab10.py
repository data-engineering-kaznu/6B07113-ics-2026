import numpy as np
import matplotlib.pyplot as plt


class HiddenMarkovModel:
    def __init__(self, states, observations, start_prob, transition_prob, emission_prob):
        self.states = states
        self.observations = observations
        self.start_prob = np.array(start_prob, dtype=float)
        self.transition_prob = np.array(transition_prob, dtype=float)
        self.emission_prob = np.array(emission_prob, dtype=float)

    def viterbi(self, observed_sequence):
        n_states = len(self.states)
        n_steps = len(observed_sequence)

        viterbi_table = np.zeros((n_states, n_steps), dtype=float)
        backpointers = np.zeros((n_states, n_steps), dtype=int)

        first_observation = self.observations.index(observed_sequence[0])
        viterbi_table[:, 0] = (
            self.start_prob * self.emission_prob[:, first_observation]
        )

        for step in range(1, n_steps):
            observation_index = self.observations.index(observed_sequence[step])
            for current_state in range(n_states):
                transition_scores = (
                    viterbi_table[:, step - 1]
                    * self.transition_prob[:, current_state]
                    * self.emission_prob[current_state, observation_index]
                )
                best_previous_state = np.argmax(transition_scores)
                viterbi_table[current_state, step] = transition_scores[best_previous_state]
                backpointers[current_state, step] = best_previous_state

        best_last_state = np.argmax(viterbi_table[:, -1])
        best_path = [best_last_state]

        for step in range(n_steps - 1, 0, -1):
            best_last_state = backpointers[best_last_state, step]
            best_path.append(best_last_state)

        best_path.reverse()
        decoded_states = [self.states[state_index] for state_index in best_path]
        final_probability = viterbi_table[best_path[-1], -1]

        return decoded_states, final_probability, viterbi_table

    def forward(self, observed_sequence):
        n_states = len(self.states)
        n_steps = len(observed_sequence)
        forward_table = np.zeros((n_states, n_steps), dtype=float)

        first_observation = self.observations.index(observed_sequence[0])
        forward_table[:, 0] = (
            self.start_prob * self.emission_prob[:, first_observation]
        )

        for step in range(1, n_steps):
            observation_index = self.observations.index(observed_sequence[step])
            for current_state in range(n_states):
                forward_table[current_state, step] = np.sum(
                    forward_table[:, step - 1] * self.transition_prob[:, current_state]
                ) * self.emission_prob[current_state, observation_index]

        return forward_table, np.sum(forward_table[:, -1])

    def plot_transition_probabilities(self, output_path):
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(self.transition_prob, cmap="Blues", vmin=0, vmax=1)

        ax.set_title("Transition probabilities")
        ax.set_xticks(np.arange(len(self.states)))
        ax.set_yticks(np.arange(len(self.states)))
        ax.set_xticklabels(self.states)
        ax.set_yticklabels(self.states)
        ax.set_xlabel("Next state")
        ax.set_ylabel("Current state")

        for row in range(len(self.states)):
            for col in range(len(self.states)):
                ax.text(
                    col,
                    row,
                    f"{self.transition_prob[row, col]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        fig.colorbar(image, ax=ax, label="Probability")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


if __name__ == "__main__":
    states = ["Sunny", "Rainy"]
    observations = ["walk", "shop", "clean"]

    start_probabilities = [0.6, 0.4]
    transition_probabilities = [
        [0.7, 0.3],
        [0.4, 0.6],
    ]
    emission_probabilities = [
        [0.6, 0.3, 0.1],
        [0.1, 0.4, 0.5],
    ]

    observed_sequence = ["walk", "shop", "clean", "clean", "walk"]

    hmm = HiddenMarkovModel(
        states=states,
        observations=observations,
        start_prob=start_probabilities,
        transition_prob=transition_probabilities,
        emission_prob=emission_probabilities,
    )

    predicted_states, path_probability, viterbi_table = hmm.viterbi(observed_sequence)
    forward_table, observation_probability = hmm.forward(observed_sequence)

    print("Observed sequence:", observed_sequence)
    print("Most probable hidden states:", predicted_states)
    print(f"Probability of the best path: {path_probability:.6f}")
    print(f"Total probability of observations: {observation_probability:.6f}")
    print("\nViterbi table:")
    print(viterbi_table)
    print("\nForward table:")
    print(forward_table)

    output_file = "transition_probabilities.png"
    hmm.plot_transition_probabilities(output_file)
    print(f"\nTransition probability chart saved to: {output_file}")
