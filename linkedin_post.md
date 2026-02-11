Your reasoning model is wasting tokens. Here's how to stop it.

I built ESTAR-JOLO - an implementation of ESTAR-LITE that watches an LLM's chain-of-thought in real time and calls "stop" when the answer has already been reached.

The "triangle angles" problem? DeepSeek-R1 used 1,522 thinking tokens. It only needed 472. That's 69% waste on a single question.

Just shipped 4 improvements that make it actually production-ready:

1. Real logprobs from Ollama - the classifier was running blind on random data. Now it reads actual token probabilities from the inference engine. The difference between a coin flip and real signal.

2. Threshold tuning script - train once, sweep (threshold x patience) combinations on your validation set. Find the accuracy/savings sweet spot without touching model code.

3. Training data diversity - combine MATH-500, GSM8K, and your own custom datasets. Better generalization across problem types.

4. Adaptive patience - short problems stop fast (patience=2), hard problems get more runway (patience=5). Formula: max(2, min(5, tokens // 100)).

Results on 10 math problems with DeepSeek-R1 1.5B running locally via Ollama:
- 1.6x token reduction
- 90% accuracy preserved
- 37% mean savings
- No retraining. No architecture changes.

The paper (ESTAR, Wang et al. 2025) reports 3.7x reduction on larger models. The gap is the logprobs - now that they're real, the next run should close it significantly.

Open source: github.com/jhammant/ESTAR-JOLO

#LLM #AI #MachineLearning #DeepLearning #Optimization #OpenSource
