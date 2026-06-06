"""Streamed-learner planted smoke for exp79 (NOT a numbered experiment).

Validates that exp79's SGNS writer + the cosine read can detect a PLANTED substitutable pair on a
corpus rich enough for distributional learning: topic-structured contexts, a planted pair (A,B) that
share a topic's context distribution but NEVER co-occur, among many diverse-context target words.
EXPECT: cos(A,B) [same-topic, zero cooc] >> cos(A, diff-topic) [random]. If SGNS can't show this, the
WikiText run is not interpretable.
"""
import importlib.util, pathlib, sys
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch

spec = importlib.util.spec_from_file_location("exp79", REPO / "experiments" / "79_betb_streamed_sgns.py")
exp79 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp79)

T, CW, NT = 12, 12, 48               # topics, context-words/topic, target words
n_ctx = T * CW                        # context-word ids 0..143
targets = list(range(n_ctx, n_ctx + NT))
V = n_ctx + NT
topic_of_target = {targets[i]: i % T for i in range(NT)}
topic_ctx = {t: list(range(t * CW, t * CW + CW)) for t in range(T)}

# plant: A,B = two targets in the SAME topic (substitutable, never co-occur by construction)
same_topic_targets = [tg for tg in targets if topic_of_target[tg] == 0]
A, B = same_topic_targets[0], same_topic_targets[1]
# a random diff-topic partner for the contrast
C = [tg for tg in targets if topic_of_target[tg] == 5][0]

g = torch.Generator().manual_seed(0)
stream = []
for _ in range(6000):                 # each "sentence": [ctx ctx TARGET ctx ctx], one target, no target-target cooc
    tg_word = targets[torch.randint(NT, (1,), generator=g).item()]
    top = topic_of_target[tg_word]
    pool = topic_ctx[top]
    left = [pool[torch.randint(CW, (1,), generator=g).item()] for _ in range(2)]
    right = [pool[torch.randint(CW, (1,), generator=g).item()] for _ in range(2)]
    stream += left + [tg_word] + right
print(f"V={V} stream_len={len(stream)} A={A} B={B}(same topic 0) C={C}(topic 5)")

centers, contexts, negp, ns = exp79.build_skipgram_pairs(stream, V, special=set(), radius=2, subsample_t=1.0)
Y = exp79.train_sgns(centers, contexts, negp, V, d=64, neg=5, epochs=12, batch=4096, lr=0.01,
                     seed=0, device="mps", learn=True)

def cos(i, j):
    return float((Y[i] * Y[j]).sum())

same = cos(A, B)
diff = cos(A, C)
# average random diff-topic pair cosine
rng = torch.Generator().manual_seed(7)
rand_cos = []
for _ in range(60):
    i, j = targets[torch.randint(NT, (1,), generator=rng).item()], targets[torch.randint(NT, (1,), generator=rng).item()]
    if i != j and topic_of_target[i] != topic_of_target[j]:
        rand_cos.append(cos(i, j))
rmean = sum(rand_cos) / len(rand_cos)
print(f"cos(A,B same-topic,zero-cooc)={same:+.3f}  cos(A,C diff-topic)={diff:+.3f}  "
      f"mean rand diff-topic={rmean:+.3f}")
print(f"SPEC (planted - random) = {same - rmean:+.3f}  "
      f"{'PASS — SGNS recovers planted substitutability' if same - rmean > 0.2 else 'FAIL'}")
