import torch

class BeamSearch:
    def __init__(self, model, x, beam_width=5):
        self.model = model
        self.x = x
        self.beam_width = beam_width

    def beam_search(self, x, max_steps=10, beam_size=5, top_k=10):
        model_output = self.model(x)
        beam = [(0, [], model_output)]  # (score, sequence, action probabilities)

        for _ in range(max_steps):
            all_candidates = []

            for score, seq, action_probs in beam:
                for action_idx in range(action_probs.size(1)):
                    action_score = action_probs[0, action_idx].item()
                    candidate_score = score + action_score
                    candidate_seq = seq + [action_idx]
                    all_candidates.append((candidate_score, candidate_seq, action_probs))

            ordered_candidates = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            beam = ordered_candidates[:self.beam_width]

        best_sequence = beam[0][1]
        return best_sequence
        