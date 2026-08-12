---
title: lecture note ideas
draft: true
tags:
---
- the learning mechanics introduction to neural networks
- toy model I: deep linear networks
	- adapt from blog post
		- get rid of all the history/literature references
		- be more verbose with math
- toy model II: kernel regression
	- why study kernel regression?
		- when lazy/ntk training happens
			- large width, large initial weights, label rescaling, output rescaling
	- how to study kernel regression?
		- start with linear regression
		- eigenlearning
	- applying eigenlearning to real datasets
	- the implications of understanding kernel regression for our understanding of deep learning
		- the generalization aspect
		- fundamental flaw: the lack of feature learning
			- neural networks in the lazy learning regime (i.e., kernel regression with the NTK) have a strong inductive bias towards functions that are aligned to the NTK (makes sense from eigenlearning perspective). so, these models will not generalize well when trained on target functions misaligned to the NTK. by contrast, neural networks in the feature learning regime can adapt their internal representations to improve the sample complexity of learning for functions that would be difficult for the NTK
		- intuition for why neural nets in the feature learning regime will outperform neural nets in the lazy regime
			- there exist settings in which feature learning models provably have better sample complexity compared to lazy models (lee et al https://arxiv.org/pdf/2206.15144)
			- The NTK has a fixed prior over functions (determined by the kernel's eigenspectrum). If the target function is not well-aligned with that prior — e.g., it lives in a low-dimensional subspace that the kernel spreads probability mass away from — then you need many samples to concentrate posterior mass on the right function.
			- Feature learning networks can _update their effective prior_ during training. The early gradient steps effectively do hypothesis selection ("this subspace of inputs matters"), and later steps do function fitting within that subspace. You're getting two things done with the same samples.
			- feature learning as kernel evolution
- the lazy and rich training regimes
	- we want to train models in the rich regime!
	- so we know that there exists a lazy regime and we know that there is a regime that is not lazy, i.e., a regime where models learn features. if there is a maximally lazy regime, is there a maximally feature learning (rich) regime?
	- use deep linear networks as a toy model to study these regimes
		- given desiderata for a feature learning model, can we determine what the hyperparameters should look like at init?
	- hp transfer
- studying models in the feature learning regime
	- for studying models in the rich feature learning regime, more or less, all bet are off; we don't have a consensus method of tackling feature learning models
		- dmft, agf, mech interp, as examples of research programs for studying models in the feature learning regime
			- different focuses: how to get stable training/how to train more efficiently? how to interpret what's going on? 


- case study I: grokking
	- example of symbiotic relationship between mech interp and lm
		- first discovered by mech interp community (power et al https://arxiv.org/pdf/2201.02177), later explained by lm community
	- precise empirics -> theory through an analytically solvable setting
		- grokking as the result of delayed feature learning
			- https://arxiv.org/pdf/2205.10343 (eric michaud): representation enables generalization
			- https://openreview.net/pdf?id=vt5mnLVIVo (blake bordelon): grokking as a transition from lazy learning to feature learning
		- balancedness -> grokking; analytically solvable toy setting
			- https://arxiv.org/pdf/2406.06158 (dan kunin)
- case study II: data statistics -> representation geometry -> computation
	- another example of symbiotic relationship between mech interp and lm