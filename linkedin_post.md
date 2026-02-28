Your reasoning model is wasting tokens. Mine was too.

I asked DeepSeek-R1 "how many degrees in a triangle?" It used 1,522 thinking tokens. It found 180 after 472. The other 1,050 tokens were the model double-checking, re-proving, and re-confirming what it already knew.

This isn't a bug. It's how reasoning models work. They think out loud and don't know when to stop.

So I built ESTAR-JOLO - a lightweight classifier that watches an LLM's chain-of-thought in real time and says "you're done" when the answer has stabilized.

How it works:

A small LightGBM model monitors four signals from the token stream at each generation step:
- Instantaneous evidence: what answer do the current token probabilities point to?
- Cumulative stability: has the leading answer been consistent, or is it still flipping?
- Early-stop curvature: is confidence converging or still moving?
- Token confidence: how sure is the model about each token it's generating?

When all signals agree the answer is locked in (threshold 0.9, sustained for 3 steps), it injects </think> and lets the model produce its final answer.

No fine-tuning. No architecture changes. No retraining. Just an external classifier reading logprobs.

Results on 10 math problems with DeepSeek-R1 1.5B running locally on my Mac via Ollama:
- 1.6x token reduction
- 90% accuracy preserved
- 37% mean savings
- Best case: 69% fewer tokens (triangle angles)

The pattern is clear: problems where the model overthinks the most get the biggest savings. Easy problems that are already fast get left alone.

The implementation includes real logprob extraction from Ollama, a threshold/patience tuning script to find your optimal stopping criteria, support for combining multiple training datasets, and adaptive patience that scales with problem difficulty.

Based on the ESTAR paper (Wang et al. 2025) which reports 3.7x reduction on larger models. This is an independent open-source implementation of the ESTAR-LITE component.

github.com/jhammant/ESTAR-JOLO

#LLM #AI #MachineLearning #DeepLearning #Optimization #OpenSource
