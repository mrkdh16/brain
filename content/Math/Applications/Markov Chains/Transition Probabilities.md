---
title: Transition Probabilities
draft: false
tags:
  - math
  - computer-science
---
 
### Theorem.
The $m$ step transition probability $P(X_{n+m} = j \mid X_n = i)$ is the $m$th power of the transition matrix $p$.
### Proof
:O

---
This is a go-to application of [[Diagonalizability|diagonalization]] and [[Linear Algebra Course Notes|linear algebra]] in general, and for good reason. It shows the immense utility of being able to easily raise some matrix to the $m$th power and why representing stochastic processes as matrices is so useful. Framing a problem in terms of linear algebra allows us to use a vast array of algorithmic and numerical tools that have been proven to work mathematically.