---
title: Linear Algebra Course Notes
draft: false
tags:
  - math
  - linear-algebra
---
Linear algebra is arguably one of the most widely applied fields of math, especially in an age where data science is practically everywhere. However, at its core, linear algebra has beautiful algebraic structure and gives rise to some very elegant mathematics. 

Below is a compilation of the most interesting results we talked about in our linear algebra class (mostly following Linear Algebra Done Right by Axler). Some highlights that I personally liked: [[Column Rank = Row Rank]], [[Euclidean Rings]], [[Riesz Representation Theorem]], [[Real Spectral Theorem]], and [[Polar Decomposition]].

[[Review 2.pdf|This]] is the review sheet that I made for my class' final exam. It contains most of the interesting results and ideas.
### Basics.
Linear algebra is ultimately about [[Vector Spaces]], essentially a special type of abelian group (groups with addition). One of the most fundamental results of linear algebra is the existence of [[Complements]]. This result can be used to derive [[The Rank Nullity Theorem]]. Another way to think of complementary spaces is through [[Quotient Spaces]]. Matrices are a handy way to think of linear maps. [[Exact Sequences]] seem to be important enough to warrant a mention.
### Duality
[[Duality]] is an idea that pops up everywhere in math. A consequence of duality is [[Column Rank = Row Rank]].
### Polynomials, Eigenvalues and vectors
The ring of polynomials, and in general, [[Euclidean Rings]] have some nice properties. [[The Minimal Polynomial]] relates any linear map to a corresponding monic polynomial. Using our knowledge of polynomials, we can use the minimal polynomial as a bridge to study linear maps.
### Jordan Form
[[Diagonalizability]] is of great interest due to ease with which we can work with diagonal matrices. Few linear operators are diagonalizable, but every operator has a Jordan Canonical Form.
### Inner Products
[[Inner Products]] brings geometry into the picture and provides an enlightening way to view vector spaces. [[Inequalities]] are used everywhere in math and many important ones are derived in linear algebra. The [[Riesz Representation Theorem]] provides a canonical isomorphism between an inner product space and its dual space and serves as a bridge between linear algebra and analysis. [[Schur's Theorem]] shows that every linear operator is upper-triangular in some orthonormal basis, and this is immensely useful.
### Spectral Theorem
The [[Real Spectral Theorem]] and [[Complex Spectral Theorem]] shows that symmetric and normal matrices respectively are diagonalizable. Spectral theorems make physicists' lives much easier.
### Decompositions
There is a good analogy to be made between complex numbers and linear maps, culminating in the [[Polar Decomposition]]. Perhaps the most immediately applicable and useful concept from our class: [[Singular Value Decomposition]].
### Tensor Products
yeah idk