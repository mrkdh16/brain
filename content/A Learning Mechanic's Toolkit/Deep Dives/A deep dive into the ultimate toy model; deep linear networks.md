---
title: A deep dive into the ultimate toy model; deep linear networks
draft: false
tags:
  - computer-science
  - linear-algebra
  - machine-learning
  - math
  - paper
  - learning-mechanics
---
<center>
<img src="Screenshot 2026-02-14 at 12.23.24 AM.png" width="600">
</center>

An interactive, albeit more concise, version of this blog post is hosted at [learningmechanics.pub](https://learningmechanics.pub/).
## Introduction.
Understanding deep learning is a *very* hard problem. Despite the rapid increase in AI capabilities, we have yet to create a cohesive mathematical framework that explains *what* and *how* these powerful models learn. When tackling a problem this monumental—specifically, the task of erecting a rigorous mathematical framework that describes a ridiculously complex system—it helps to think like a physicist. Likely, a physicist's first instinct would be to conjure up a toy model. By studying a simplified system that still exhibits the complex, non-trivial phenomena of the original, we can find mathematically solid ground.

At first glance, deep linear networks seem quite uninteresting. Indeed, no expressiveness is gained from adding layers in linear networks as the input-output map can always be rewritten as a single shallow layer:
$$
\hat{y}= W_{l}W_{l-1} \ldots W_{1}x = W_{\text{total}}x.
$$
However, their training dynamics are nonlinear and thus capture some of the mysterious phenomena that we observe in their nonlinear cousins. Such phenomena include long plateaus at saddle points, rapid drops in the loss and low-rank bias for solutions. Deep linear networks are a rare breed of highly mathematically tractable models that still retain interesting phenomena. They offer a golden opportunity to probe both the *what* and *how* in an analytically tractable setting. We should not take this for granted.

We shall follow the work of [Saxe et al. (2014)](https://arxiv.org/pdf/1312.6120) and [Saxe et al. (2018)](https://arxiv.org/pdf/1810.10531). The first paper, titled "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks," is the more 'canonical text' within the deep learning community. It focuses on the derivation of the theory and various implications related to machine learning. The second paper, titled "A mathematical theory of semantic development in deep neural networks," seems to aim for a broader audience with a focus on 'semantic cognition.' Figures are pulled from both papers: figure 1 and 3 from the latter, and figure 4 from the former. Figure 2 was generated from [this](https://alexlenail.me/NN-SVG/) website.
## Setup.
#### A toy task for a toy model.
It will prove useful to keep in mind a simple toy task. Imagine the network is presented with an item $\mu$ (e.g. a "Canary") represented as a one-hot input vector $x^\mu$[^1]. The network's objective is to predict a vector of features $y^\mu$, such as "Can Fly," "Has Wings," or "Is Yellow".

Throughout training, the network experiences many such examples $\{x^\mu, y^\mu\}$. The statistical structure of this environment is captured by the input-output correlation matrix $\Sigma_{xy} \equiv \sum_{\mu}y^\mu {x^\mu}^\top$:

<center>
<img src="Screenshot 2026-02-12 at 9.48.02 PM.png" width="200">
<figcaption>Figure 1: Example of an input-output correlation matrix</figcaption>
</center>

This matrix represents how strongly specific items correlate with specific properties across the entire dataset and is a key component of our analysis.
#### The toy model.
Consider a simple 2-layer linear network $\hat{y} = W_2 W_1 x$ with training data $\{ x^{\mu},y^\mu\}$ ($\mu=1,\dots,P$) and mean squared error $\mathcal{E} = \sum^P_{\mu=1}\frac{1}{2}||y^\mu-W_2 W_1 x^\mu||^2_2$. Let's say that the input has dimension $N_1$, the hidden layer has dimension $N_{2}$ and the output has dimension $N_{3}$. So, $x^\mu \in \mathbb{R}^{N_{1}}$, $y^\mu \in \mathbb{R}^{N_{3}}$, $W_1$ is an $N_{2}\times N_{1}$ matrix and $W_2$ is an $N_{3}\times N_{2}$ matrix[^2]. In terms of our toy task, this means that we have $N_1$ items, $N_3$ possible features that the items can have and $P$ examples of items having features.

<center>
<img src="Screenshot 2026-02-14 at 12.39.03 AM.png" width="350">
<figcaption>Figure 2: A 2-layer network</figcaption>
</center>

#### Taking derivatives.
To run gradient descent on $\mathcal{E}$, we need to compute the partial derivatives for each of the weight matrices. The simplest way to do this is how a computer does it: with backpropagation (see [[2-layer linear network backprop]] for derivation). We get the following partial derivatives from backprop:
$$
\begin{align*}
\frac{\partial \mathcal{E}}{\partial W_1} &= \sum^P_{\mu=1} \frac{\partial}{\partial W_1} \frac{1}{2}||y^\mu - W_2 W_1 x^\mu||^2_{2}  \\
&= \sum^P_{\mu=1} -{W_2}^\top (y^\mu {x^\mu}^\top - W_2 W_1 x^\mu {x^\mu}^\top) \\
&= -{W_2}^\top \left( \sum^P_{\mu=1}y^\mu {x^\mu}^\top-W_2 W_1\sum^P_{\mu=1}x^\mu {x^\mu}^\top \right) \\
&= -{W_2}^\top (\Sigma_{xy}-W_2 W_1\Sigma_{xx})
\end{align*}
$$
$$
\begin{align*}

\frac{\partial \mathcal{E}}{\partial W_2} &= \sum^P_{\mu=1} \frac{\partial}{\partial W_2} \frac{1}{2}||y^\mu - W_2 W_1 x^\mu||^2_{2} \\

&= \sum^P_{\mu=1} -(y^\mu {x^\mu}^\top - W_2 W_1 x^\mu {x^\mu}^\top){W_1}^\top \\

&= -\left( \sum^P_{\mu=1}y^\mu {x^\mu}^\top  - W_2 W_1\sum^P_{\mu=1}x^\mu {x^\mu}^\top  \right){W_1}^\top \\

&= -(\Sigma_{xy}-W_2 W_1\Sigma_{xx}){W_1}^\top

\end{align*}
$$
where we defined the input-output correlation matrix as $\Sigma_{xy}\equiv \sum^P_{\mu=1}y^\mu {x^\mu}^\top$ and the input correlation matrix as $\Sigma_{xx} \equiv \sum^P_{\mu=1} x^\mu {x^\mu}^\top$. Observe that the matrices $\Sigma_{xy}$ and $\Sigma_{xx}$ contain all the information from the dataset used in training.

To use tools from differential equations, we'll have to find the 'gradient flow' of the weights. We start by examining how the weight matrices get updated:
$$
\begin{align*}
\Delta W_1 &= -\lambda\frac{\partial \mathcal{E}}{\partial W_1}=  \lambda{W_2}^\top (\Sigma_{xy}-W_2 W_1\Sigma_{xx})
\\
\Delta W_2 &= -\lambda\frac{\partial \mathcal{E}}{\partial W_2}= \lambda (\Sigma_{xy}-W_2 W_1\Sigma_{xx}){W_1}^\top
\end{align*}
$$
where $\lambda$ is some small positive number, i.e. the learning rate. In the continuous-time limit, i.e. gradient flow, we can replace the discrete updates with continuous time derivatives:
$$
\begin{align}
\frac{dW_1}{dt} = {W_2}^\top (\Sigma_{xy}-W_2 W_1\Sigma_{xx}) \\
\frac{dW_2}{dt} = (\Sigma_{xy}-W_2 W_1\Sigma_{xx}){W_1}^\top. \\
\end{align}
$$
From these gradient flow equations we can show that $\frac{d}{dt}(W_2^\top W_2 - W_1 W_1^\top) = 0$, i.e., the difference $W_2^\top W_2 - W_1 W_1^\top$ is a conserved quantity throughout training (see [[Balancedness and lazy rich learning]] for the full derivation). This will become useful later.
#### What the network learns.
Looking at equations (1) and (2), we can already discern what the network will learn. The weights will stop changing and will have reached a minima of the loss when $\frac{dW_1}{dt}$, $\frac{dW_2}{dt} \approx 0$, i.e. when $W_2 W_1\Sigma_{xx} \approx \Sigma_{xy}$. In other words, the network will learn weight matrices $W_2$ and $W_1$ such that $W_2 W_1\Sigma_{xx} \approx \Sigma_{xy}$. Intuitively, the deviation of $W_2 W_1\Sigma_{xx}$ from $\Sigma_{xy}$ is what generates the learning signal. In this sense, $\Sigma_{xy}$ is the 'target' the network is chasing. On the other hand, $\Sigma_{xx}$  only multiplicatively modifies the model matrix $W_2 W_1$, and it doesn't directly drive the gradient. This is most apparent when the weight matrices $W_2$ and $W_1$ are vanishingly small (which would be the case at initialization if we initialized small). When the weights are very small, the entire $W_2 W_1\Sigma_{xx}$ term is approximately equal to 0, so the gradient is almost entirely dictated by $\Sigma_{xy}$.

That the input-output correlation matrix $\Sigma_{xy}$ generates the learning signal is an important intuition to have in mind. This matrix captures the important structure in the data and will dictate the coordinate system we work in.
#### Plan of Attack.
Equations (1) and (2) are quite complex, containing interconnected, nonlinear interactions. We are dealing with a coupled system in the sense that a change in one weight (an entry in $W_1$ or $W_2$) will affect every other weight. This kind of coupling is the main reason why studying deep networks is usually intractable. An exception to this intractability occurs when we are able to find a *decoupled* regime.

What does it mean for a regime to be decoupled? In this case, we want to find a *coordinate system* in which the interdependent matrix dynamics separate into simpler independent one-dimensional dynamics. Mathematically, this means that the change in a specific weight depends only on itself and the relevant statistical mode of the data, rather than the state of the entire weight matrix. By identifying these independent "modes" of learning, we can reduce a complex system of intertwined differential equations into a collection of parallel scalar equations. Once we have parallel, decoupled scalar differential equations, we will be able to *solve* for the dynamics, i.e. extract the full learning trajectory.

The concrete steps we will take are as follows:
1. **Identify the simplifying coordinate system**. We want to work in a fixed coordinate system that reflects the structure of the data. The two natural potential candidates are the singular vectors of the input-output correlation matrix ($\Sigma_{xy}=\Sigma_{\mu}y^\mu {x^\mu}^\top$) and of the input-input correlation matrix ($\Sigma_{xx}=\Sigma_{\mu}x^\mu {x^\mu}^\top$). Recall that $\Sigma_{xy}$ is what generates the learning signal. $\Sigma_{xx}$ is more of a '[preconditioner](https://rkube.github.io/jekyll/update/2021/03/02/neural-networks-and-iterative-solvers.html)' for the network matrix. This is intuitively why the singular vectors of $\Sigma_{xy}$ are the natural coordinate axes. To eliminate the competition between the two natural bases, we will assume $\Sigma_{xx} = I$. Notice that once $\Sigma_{xx} = I$, the gradient flow equations (1) and (2) immediately reveal that the learning task is matrix factorization: the network must find $W_2$ and $W_1$ such that $W_2 W_1 \approx \Sigma_{xy}$.
2. **Reduce to a growth-only picture**. Matrix factorization in principle involves two subtasks: *aligning* the network's principal directions with the singular vectors of $\Sigma_{xy}$, and *growing* the effective singular values to match the data's. We re-express the dynamics in the SVD basis of $\Sigma_{xy}$ to make this decomposition explicit, then assume that alignment happens extremely quickly and focus on the dynamics after this alignment phase. This reduces the coupled matrix dynamics to independent scalar equations—one per singular value.
It turns out that this growth-only picture is both a simple way to view learning and a good approximation of what happens empirically. We will also show why quick alignment is actually quite reasonable.
#### The simplifying assumption necessary for step 1.
Let us make our first simplifying assumption: that the covariance of the input vectors $x^\mu$ is whitened, i.e. that $\Sigma_{xx}=I$. Back when the original Saxe et al. (2014) paper was published, this was a reasonable assumption as it was often done in practice. This is no longer the case and that $\Sigma_{xx}=I$ is more of an assumption born out of the desire for an analytically tractable system. In fact, without this assumption, [the problem is NP-hard](https://arxiv.org/abs/1012.0197).

Once we assume that $\Sigma_{xx}=I$, the input-output correlation matrix $\Sigma_{xy}$ becomes the sole source of information about the training data that can be used in learning. This assumption simplifies equations (1) and (2):
$$
\begin{align}
\frac{dW_1}{dt} &= {W_2}^\top (\Sigma_{xy}-W_2 W_1) \\
\frac{dW_2}{dt} &=  (\Sigma_{xy}-W_2 W_1){W_1}^\top. \\
\end{align}
$$
The problem has been reduced to matrix factorization; the network will learn weight matrices $W_2$ and $W_1$ such that $W_2 W_1 \approx \Sigma_{xy}$.

Even after this simplifying assumption, we are left with complicated matrix expressions. Might there be a way to get vector, or perhaps even scalar equations that are more easily solvable? It turns out the answer is yes (with some more simplifying assumptions), but to get there, we first need to take a closer look at the input-output correlation matrix.
#### A closer look at the natural coordinate system.
Recall from the plan of attack that we want the singular vectors of $\Sigma_{xy}$ to determine the coordinate system we work in. Before we proceed, let us explore the significance of these singular vectors.

Consider the (compact) [singular value decomposition (SVD)](https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/book-chapter-4.pdf) of $\Sigma_{xy}$:
$$
\Sigma_{xy} = U S_* V^\top = \sum_{\alpha=1}^{r} s_\alpha \mathbf{u}_\alpha {\mathbf{v}_{\alpha}}^\top.
$$
where $r$ is the rank of $\Sigma_{xy}$. Here $\alpha$ indexes the modes[^3], $\mathbf{u}_\alpha$ and $\mathbf{v}_\alpha$ are the left and right singular vectors (columns of $U$ and $V$), and $s_\alpha$ are the singular values (diagonal entries of $S_*$).

<center>
<img src="Screenshot 2026-02-12 at 10.04.20 PM.png" width=700>
<figcaption>Figure 3: Modes link a set of coherently covarying properties with a set of coherently covarying items</figcaption>
</center>

Using the SVD allows us to probe the data in terms of independent "modes." We refer to the columns of $V$ (i.e. the rows of $V^\top$) as *input-analyzing* vectors, or *input modes*. The $\alpha$th input mode determines the position of any given input item along an important 1-D semantic dimension. As a concrete example, consider the second row vector of $V^\top$ in figure 3. This vector, corresponding to the second input mode, acts as an "animal-plant" axis. Along this axis, animals (Canary, Salmon) have positive values while plants (Oak, Rose) have negative values. This animal-plant axis is a way of categorizing the data determined by mathematical structure.

We refer to the columns of $U$ as *output-analyzing* vectors, or *output modes*. Similar to input modes, the $\alpha$th output mode determines the position of any given output feature along an important 1-D semantic dimension. Every input mode has a corresponding output mode and vice versa. In figure 3, the second column vector of $U$ ($\mathbf{u}_2$) pairs with the second row vector of $V^\top$ ($\mathbf{v}_2^\top$) to define a unified animal-plant axis[^4]. While the input mode determines which items belong to the category (animals v.s. plants in this case), the output mode determines which properties belong to the category. Along this second output mode axis, more animal-like properties will have higher values, while more plant-like properties will have lower values (roots < leaves, petals < fly, swim < move). The "categorizing power" of any given axis is quantified by its singular value $s_\alpha$. Larger singular values correspond to more important distinctions, while smaller singular values correspond to finer subordinate details.
#### From matrix to vector equations.
We use the input and output modes to rotate into the natural coordinate system we want to work in:
$$
\tilde{W}_1 \equiv W_1 V, \space \tilde{W}_2 \equiv U^\top W_2.
$$
We denote by $\mathbf{a}_\alpha$ the $\alpha$th column vector of $\tilde{W}_1$ and by $\mathbf{b}_\alpha$ the $\alpha$th row vector of $\tilde{W}_2$ (both $\mathbf{a}_\alpha = \mathbf{a}_\alpha(t)$ and $\mathbf{b}_\alpha = \mathbf{b}_\alpha(t)$ are time dependent). We refer to these vectors $\mathbf{a}_\alpha$ and $\mathbf{b}_\alpha$ as *connectivity modes*[^5] since they connect input and output modes to hidden neurons.

Inserting our new matrices into equations (3) and (4),
$$
\begin{align*} \frac{d W_1}{dt} &= \tilde{W}_2^\top U^\top \left( U S_* V^\top - U \tilde{W}_2 \tilde{W}_1 V^\top \right) \\
&= \tilde{W}_2^\top \left( S_* - \tilde{W}_2 \tilde{W}_1 \right) V^\top \\ \\
\implies  \frac{d W_1}{dt} V =  \frac{d \tilde{W}_1}{dt} &=\tilde{W}_2^\top \left( S_* - \tilde{W}_2 \tilde{W}_1 \right) \\\\
 \frac{d\tilde{W}_2}{dt} &= \left( U S_* V^\top - U \tilde{W}_2 \tilde{W}_1 V^\top \right) V \tilde{W}_1^\top \\
&= U \left( S_* - \tilde{W}_2 \tilde{W}_1 \right) \tilde{W}_1^\top \\ \\ \implies  U^\top \frac{d W_2}{dt}
=  \frac{d \tilde{W}_2}{dt} &= \left( S_* - \tilde{W}_2 \tilde{W}_1 \right) \tilde{W}_1^\top \end{align*}
$$
we get new equations that can be thought of in terms of vector equations. Let's first examine the time derivative of $\tilde{W}_1$:
$$
 \frac{d}{dt} \tilde{W}_1 =
\underbrace{\begin{bmatrix} | & & | \\ \frac{d\mathbf{a}_1}{dt} & \cdots & \frac{d\mathbf{a}_{N_1}}{dt} \\ | & & | \end{bmatrix} }_{[N_{2} \times N_{1}]}

= \underbrace{\begin{bmatrix} | & & | \\ \mathbf{b}_1 & \cdots & \mathbf{b}_{N_3} \\ | & & | \end{bmatrix}}_{[N_{2} \times N_{3}]}
\left(
\underbrace{\begin{bmatrix} s_1 & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & s_{N_1} \\ \vdots & \ddots & \vdots \\ 0 & \cdots & 0 \end{bmatrix}}_{[N_{3}\times N_{1}]}
-
\underbrace{\begin{bmatrix} \text{---} & \mathbf{b}_1 & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{b}_{N_3} & \text{---} \end{bmatrix}}_{[N_{3}\times N_{2}]}
\underbrace{\begin{bmatrix} | & & | \\ \mathbf{a}_1 & \cdots & \mathbf{a}_{N_1} \\ | & & | \end{bmatrix}}_{[N_{2}\times N_{1}]} \right).
$$
Looking at a single column vector $\mathbf{a}_\alpha$ of $\tilde{W}_1$,
$$
 \frac{d}{dt}\mathbf{a}_\alpha =

 \underbrace{\begin{bmatrix} | \\ \frac{d\mathbf{a}_\alpha}{dt} \\ | \end{bmatrix}}_{[N_{2}\times 1]} =
\underbrace{\begin{bmatrix} | & & | \\ \mathbf{b}_1 & \cdots & \mathbf{b}_{N_3} \\ | & & | \end{bmatrix}}_{\tilde{W}_2^\top}
\left( \underbrace{ \begin{bmatrix} 0 \\ \vdots \\ s_\alpha \\ \vdots \\ 0 \end{bmatrix} }_{\text{Column } \alpha \text{ of } S_*} - \underbrace{ \begin{bmatrix} \text{---} & \mathbf{b}_1 & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{b}_{N_3} & \text{---} \end{bmatrix} \begin{bmatrix} | \\ \mathbf{a}_\alpha \\ | \end{bmatrix} }_{\text{Column } \alpha \text{ of } \tilde{W}_2\tilde{W}_1} \right)
$$
<details markdown="1"> <summary> Detailed derivation  </summary>

$$
\begin{align*}
&=
\begin{bmatrix} | & & | \\ \mathbf{b}_1 & \cdots & \mathbf{b}_{N_3} \\ | & & | \end{bmatrix} \left( \begin{bmatrix} 0 \\ \vdots \\ s_\alpha \\ \vdots \\ 0 \end{bmatrix} - \begin{bmatrix} \mathbf{b}_1 \cdot \mathbf{a}_\alpha \\ \vdots \\ \mathbf{b}_\alpha \cdot \mathbf{a}_\alpha \\ \vdots \\ \mathbf{b}_{N_3} \cdot \mathbf{a}_\alpha \end{bmatrix} \right)
\\\\
&=
\underbrace{\begin{bmatrix} \vert & &\vert && | \\ \mathbf{b}_1 & \cdots & \mathbf{b}_\alpha & \cdots & \mathbf{b}_{N_3} \\ | & &  | & & | \end{bmatrix}}_{[N_{2} \times N_{3}]} \underbrace{\begin{bmatrix} - (\mathbf{b}_1 \cdot \mathbf{a}_\alpha) \\ \vdots \\ (s_\alpha - \mathbf{b}_\alpha \cdot \mathbf{a}_\alpha) \\ \vdots \\ - (\mathbf{b}_{N_3} \cdot \mathbf{a}_\alpha) \end{bmatrix}}_{[N_{3}\times 1]}
\end{align*}
$$
</details>

$$
=
\mathbf{b}_\alpha (s_\alpha - \mathbf{b}_\alpha \cdot \mathbf{a}_\alpha) - \sum_{\gamma \neq \alpha} \mathbf{b}_\gamma (\mathbf{b}_\gamma \cdot \mathbf{a}_\alpha).
$$

Now turning to the time derivative of $\tilde{W}_2$,
$$
 \frac{d}{dt} \tilde{W}_2 =   \underbrace{\begin{bmatrix} \text{---} & d\mathbf{b}_1/dt & \text{---} \\ & \vdots & \\ \text{---} & d\mathbf{b}_{N_3}/dt & \text{---} \end{bmatrix}}_{[N_{3}\times N_{2}]} =
\left(
\underbrace{\begin{bmatrix} s_1 & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & s_{N_1} \\ \vdots & \ddots & \vdots \\ 0 & \cdots & 0 \end{bmatrix}}_{[N_{3}\times N_{1}]}
- \underbrace{ \begin{bmatrix} \text{---} & \mathbf{b}_1 & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{b}_{N_3} & \text{---} \end{bmatrix} }_{[N_3 \times N_2]} \underbrace{ \begin{bmatrix} | & & | \\ \mathbf{a}_1 & \cdots & \mathbf{a}_{N_1} \\ | & & | \end{bmatrix} }_{[N_2 \times N_1]} \right) \underbrace{ \begin{bmatrix} \text{---} & \mathbf{a}_1 & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{a}_{N_1} & \text{---} \end{bmatrix} }_{[N_1 \times N_2]}
$$
This time looking at a single row vector $\mathbf{b}_\alpha$ of $\tilde{W}_2$,
$$
 \frac{d}{dt} \mathbf{b}_\alpha=
\underbrace{ \begin{bmatrix} \text{---} \frac{d\mathbf{b}_\alpha}{dt} \text{---} \end{bmatrix} }_{[1 \times N_{{2}}]}
=
\left( \underbrace{ \begin{bmatrix} 0 & \cdots  & s_\alpha & \cdots & 0\end{bmatrix} }_{\text{Row } \alpha \text{ of } S_*} -
\underbrace{ \begin{bmatrix} \text{---} & \mathbf{b}_\alpha & \text{---}  \end{bmatrix} \begin{bmatrix} | & & | \\ \mathbf{a}_1 & \cdots & \mathbf{a}_{N_1} \\ | & & | \end{bmatrix} }_{\text{Row } \alpha \text{ of } \tilde{W}_2\tilde{W}_1} \right)
\underbrace{ \begin{bmatrix} \text{---} & \mathbf{a}_1 & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{a}_{N_1} & \text{---} \end{bmatrix}}_{\tilde{W}_1^\top}
$$
<details markdown="1"> <summary>Detailed derivation</summary>

$$
\begin{align*}
&=
\left( \begin{bmatrix} 0 & \cdots  & s_\alpha & \cdots & 0\end{bmatrix} - \begin{bmatrix} \mathbf{b}_\alpha \cdot \mathbf{a}_1 & \cdots  & \mathbf{b}_\alpha \cdot \mathbf{a}_\alpha & \cdots & \mathbf{b}_\alpha \cdot \mathbf{a}_{N_{1}}\end{bmatrix}
\right)\begin{bmatrix} \text{---} & \mathbf{a}_1 & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{a}_{N_1} & \text{---} \end{bmatrix}
\\\\
&=
\underbrace{\begin{bmatrix} -(\mathbf{b}_\alpha \cdot \mathbf{a}_1) & \cdots  & (s_\alpha-\mathbf{b}_\alpha \cdot \mathbf{a}_\alpha) & \cdots & -(\mathbf{b}_\alpha \cdot \mathbf{a}_{N_{1}})\end{bmatrix}}_{1 \times N_{1}}
\underbrace{\begin{bmatrix} \text{---} & \mathbf{a}_1 & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{a}_\alpha & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{a}_{N_1} & \text{---} \end{bmatrix}}_{[N_{1}\times N_{2}]}
\end{align*}
$$

</details>

$$
=
(s_\alpha - \mathbf{b}_\alpha \cdot \mathbf{a}_\alpha)\mathbf{a}_{\alpha} - \sum_{\gamma \neq \alpha} (\mathbf{b}_\alpha \cdot \mathbf{a}_\gamma)\mathbf{a}_\gamma.
$$
Thus, we have successfully turned our matrix equations (3) and (4) into vector equations:
$$
\begin{align}
 \frac{d}{dt}\mathbf{a}_\alpha &= \mathbf{b}_\alpha (s_\alpha - \mathbf{b}_\alpha \cdot \mathbf{a}_\alpha) - \sum_{\gamma \neq \alpha} \mathbf{b}_\gamma (\mathbf{b}_\gamma \cdot \mathbf{a}_\alpha) \\
 \frac{d}{dt} \mathbf{b}_\alpha &= (s_\alpha - \mathbf{b}_\alpha \cdot \mathbf{a}_\alpha)\mathbf{a}_{\alpha} - \sum_{\gamma \neq \alpha} (\mathbf{b}_\alpha \cdot \mathbf{a}_\gamma)\mathbf{a}_\gamma.
\end{align}
$$
Equations (5) and (6) reveal the structure of the learning problem in a useful way. Each connectivity mode pair $\mathbf{a}_\alpha, \mathbf{b}_\alpha$ is driven toward alignment by its corresponding singular value $s_{\alpha}$​ through the first term, but is simultaneously coupled to every other mode through the sum $\sum_{{\gamma\neq \alpha}}$ (essentially, the network wants to learn $\mathbf{a}_\alpha \cdot \mathbf{b}_\beta \approx s_{\alpha}\delta_{\alpha \beta}$[^6]). The cross-terms are the sole source of inter-mode dependence and are what make the system difficult to solve. However, notice that they vanish exactly when distinct connectivity modes are orthogonal, i.e. when $\mathbf{b}_\gamma \cdot \mathbf{a}_\alpha = 0$ and $\mathbf{b}_\alpha \cdot \mathbf{a}_\gamma = 0$ for all $\gamma \neq \alpha$. In other words, the matrix factorization problem $W_2 W_1 \approx \Sigma_{xy}$ that gradient descent must solve involves two subtasks: *aligning* the connectivity modes so that cross-mode interference disappears, and *growing* the effective mode strengths $\mathbf{a}_\alpha \cdot \mathbf{b}_\alpha$ toward the corresponding singular values $s_\alpha$​. Once alignment is achieved, the coupled vector equations collapse into independent scalar equations—one per mode—and we can solve for the full time course of learning.
## How the network learns.
Although we have inferred what the network will look like after it has reached a solution ($W_2 W_1 \approx \Sigma_{xy}$), we failed to derive any expression with a time dependence. This is unfortunate since we want to understand how the model will evolve throughout training. However, solving equations (5) and (6) from scratch is difficult because all the differential equations are intertwined. This is where an orthogonality assumption can be useful (see [[Quick (Silent) Alignment]] for a justification for this assumption, i.e. an argument for quick alignment early on in training).
#### From vector to scalar equations.
Let's assume that
$$
\mathbf{a}_\alpha(t) = a_{\alpha}(t) \mathbf{r}_\alpha, \space \mathbf{b}_\alpha(t) = b_{\alpha}(t)\mathbf{r}_\alpha
$$
where $a_{\alpha}(t),b_{\alpha}(t)$ are scalar variables with a time dependence and $\mathbf{r}_\alpha$ are unit vectors such that $\mathbf{r}_\alpha \cdot \mathbf{r}_\beta = \delta_{\alpha \beta}$[^7]. This means that $\mathbf{a}_\alpha \cdot \mathbf{b}_\beta = a_{\alpha}b_{\beta}\cdot\delta_{{\alpha \beta}}$. Here, we are assuming something slightly stronger than simple orthogonality of the connectivity modes[^8]: we are assuming that connectivity modes of the same index start out perfectly aligned.

Recall that matrix factorization involves two distinct subtasks: alignment and effective singular value growth. Since we've assumed alignment, the only task left is effective singular value growth. We can see this by rotating back to the original basis:
$$
\begin{align}
W_2 W_1 =U\tilde{W_{2}}\tilde{W_{1}}V^\top = U \,\text{diag}(\hat{s}_{1}(t),\dots,\hat{s}_{N_{2}}(t),0,\dots,0)\, V^\top  = U \hat{S}(t) V^\top
\end{align}
$$
where, under our alignment assumption, the $(\alpha,\beta)$ entry of $\tilde{W_{2}}\tilde{W_{1}}$ is 
$$
\mathbf{b}_{\alpha}\cdot \mathbf{a}_{\beta}=b_{\alpha}a_{\beta}\mathbf{r}_{\alpha}\cdot \mathbf{r_{\beta}} = b_{\alpha}a_{\beta}\delta_{{\alpha \beta}}
$$
which implies that $\tilde{W_{2}}\tilde{W_{1}}$ is diagonal. We define $\hat{S}(t)\equiv \tilde{W_{2}}\tilde{W_{1}}$ as the diagonal matrix with values $a_{\alpha}b_{\alpha}$ on the diagonal. We also define $\hat{s}_{\alpha}(t) \equiv a_{\alpha}(t)b_{\alpha}(t)$ as the 'effective singular value' (equivalently, the 'effective mode strength') of the network at time $t$. In this decoupled regime, gradient descent on the network reduces to growing the effective singular values $\hat{s}_{\alpha}$ toward $s_{\alpha}$ assuming we start with small $a_{\alpha}$ and $b_{\alpha}$.

Inserting our new variables into equations (5) and (6),
$$
\begin{align*}
 \frac{d}{dt}a_{\alpha}\mathbf{r}_\alpha &= b_{\alpha}\mathbf{r}_\alpha (s_\alpha - b_{\alpha}\mathbf{r}_\alpha \cdot a_{\alpha}\mathbf{r}_\alpha) - \sum_{\gamma \neq \alpha} b_{\gamma}\mathbf{r}_\gamma (b_{\gamma}\mathbf{r}_\gamma \cdot a_{\alpha}\mathbf{r}_\alpha) \\
&=b_{\alpha}(s_{\alpha} - b_{\alpha}a_{\alpha})\mathbf{r}_\alpha \\
\implies  \frac{d}{dt}a_{\alpha} &= b_{\alpha}(s_{\alpha}-a_{\alpha}b_{\alpha})
\\\\
 \frac{d}{dt}b_{\alpha}\mathbf{r}_\alpha &= (s_\alpha - b_{\alpha}\mathbf{r}_\alpha \cdot a_{\alpha}\mathbf{r}_\alpha) a_{\alpha}\mathbf{r}_\alpha - \sum_{\gamma \neq \alpha} (b_{\alpha}\mathbf{r}_\alpha \cdot a_{\gamma}\mathbf{r}_\gamma)a_{\gamma}\mathbf{r}_\gamma \\
&=(s_{\alpha} - b_{\alpha}a_{\alpha})a_{\alpha}\mathbf{r}_\alpha \\
\implies  \frac{d}{dt}b_{\alpha} &= a_{\alpha}(s_{\alpha}-a_{\alpha}b_{\alpha}).
\end{align*}
$$
Thus, we have successfully turned our vector equations (5) and (6) into decoupled, independent scalar equations:
$$
\begin{align}
 \frac{d}{dt}a_{\alpha} &= b_{\alpha}(s_{\alpha}-a_{\alpha}b_{\alpha})  \\
 \\ \frac{d}{dt}b_{\alpha} &= a_{\alpha}(s_{\alpha}-a_{\alpha}b_{\alpha}).
\end{align}
$$
From equations (8) and (9) we can infer a conservation law: $\frac{d}{dt}(a_\alpha^2-b_\alpha^2)=0$. This is the scalar analogue of the conserved quantity $W_{1}W_{1}^\top-W^\top_{2}W_{2}$ which we derived from the original gradient flow equations (1) and (2).

We need to make one last assumption to create an easily solvable differential equation: that $a_{\alpha}(0) \approx b_{\alpha}(0)$ at initialization. This is a reasonable assumption if we assume $a_{\alpha},b_{\alpha}$ are both initialized with very small values. Since we know that the quantity $a^2_{\alpha}-b^2_{\alpha}$ stays constant throughout training, an assumption that $a_{\alpha}(0) \approx b_{\alpha}(0)$ at initialization is enough to ensure that $a_{\alpha}(t) \approx b_{\alpha}(t)$ throughout training.
#### A differential equation we can solve.
Consider the derivative of the effective singular value $\hat{s}_{\alpha}(t)=a_{\alpha}(t)b_{\alpha}(t)$ using the product rule:
$$
\begin{align*}
 \frac{d\hat{s}_{\alpha}}{dt} =  \left( b_{\alpha} \frac{da_{\alpha}}{dt} + a_{\alpha} \frac{db_{\alpha}}{dt} \right) &= b_{\alpha}^2(s_{\alpha}-a_{\alpha}b_{\alpha}) + a_{\alpha}^2(s_{\alpha}-a_{\alpha}b_{\alpha}) \\
&= (s_{\alpha}-a_{\alpha}b_{\alpha})(a_{\alpha}^2+b_{\alpha}^2) \\
&\approx 2a_{\alpha}b_{\alpha}(s_{\alpha}-a_{\alpha}b_{\alpha}) \\
&= 2\hat{s}_{\alpha}(s_{\alpha}-\hat{s}_{\alpha})
\end{align*}
$$
where we used the assumption that $a\approx b$ to get $a^2+b^2\approx 2ab$. Now,
$$
\begin{align}
 \frac{d\hat{s}_{\alpha}}{dt} = 2\hat{s}_{\alpha}(s_{\alpha}-\hat{s}_{\alpha})
\end{align}
$$
is a separable differential equation that we can easily solve:
$$
\begin{align*}
\int_{{\hat{s}_{\alpha}}^0}^{\hat{s}_{\alpha}(t)} \frac{1}{2\hat{s}(s_{\alpha}-\hat{s})} d\hat{s} &= \int_{t_0}^{t} dt' \\\\

\implies \Delta t &=  \int_{{\hat{s}_{\alpha}}^0}^{\hat{s}_{\alpha}(t)} \frac{1}{-2\hat{s}^2 + 2\hat{s}s_{\alpha}} d\hat{s} \\\\
&= \frac{1}{2s_{\alpha}} \ln \frac{\hat{s}_{\alpha}(t)(s_{\alpha}-{\hat{s}_{\alpha}}^0)}{{\hat{s}_{\alpha}}^0(s_{\alpha}-\hat{s}_\alpha(t))}.
\end{align*}
$$
Where we can use partial fraction decomposition to solve the integral.

Suppose $\hat{s}_{\alpha}$ starts at a small value i.e. $\hat{s}_{\alpha}^0 = \hat{s}_{\alpha}(t_{0}) =\varepsilon$ and $t_{0}=0$ . How long does it take for $\hat{s}_{\alpha}(t)$ to equal $s_{\alpha}-\varepsilon$?
$$
\begin{align*}
t &= \frac{1}{2s_{\alpha}} \ln{\frac{(s_{\alpha}-\varepsilon)^2}{\varepsilon^2}} \\
&= \frac{1}{s_{\alpha}} \ln \frac{s_{\alpha}-\varepsilon}{\varepsilon} \\
&= O\left( \frac{1}{s_{\alpha}} \right)
\end{align*}
$$
Thus, the amount of time it takes for the effective singular value $\hat{s}_{\alpha}$ to grow to $s_{\alpha}$ for small initialization is inversely proportional to $s_{\alpha}$. The network learns larger singular values corresponding to more significant input-output modes first.

If we assume that $t_{0}=0$, then by rearranging terms,
$$
\begin{align*}
2ts_{\alpha} &= \ln{\frac{\hat{s}_\alpha(t)(s_{\alpha}-{\hat{s}_{\alpha}}^0)}{{\hat{s}_{\alpha}}^0(s_{\alpha}-\hat{s}_\alpha(t))}}\\
\implies e^{2ts_{\alpha}}&= \frac{\hat{s}_\alpha(t)(s_{\alpha}-{\hat{s}_{\alpha}}^0)}{{\hat{s}_{\alpha}}^0(s_{\alpha}-\hat{s}_\alpha(t))} \\
\implies \hat{s}_\alpha(t)(s_{\alpha}-{\hat{s}_{\alpha}}^0+{\hat{s}_{\alpha}}^0e^{2s_{\alpha}t}) &={\hat{s}_{\alpha}}^0 s_{\alpha}e^{2s_{\alpha}t}\\\\
\implies \hat{s}_\alpha(t) = a_{\alpha}(t)b_{\alpha}(t)&= \frac{s_{\alpha}e^{2s_{\alpha}t}}{e^{2s_{\alpha}t}-1+\frac{s_{\alpha}}{{\hat{s}_{\alpha}}^0}} \\
&=\frac{s_{\alpha}}{1+\left( \frac{s_{\alpha}}{{\hat{s}_{\alpha}}^0}-1 \right){e^{-2s_{\alpha}t}}}.
\end{align*}
$$
Thus, we have found an expression for how the effective singular values will evolve throughout training. This is a significant result since through equation (7), we are able to characterize the evolution of all the parameters of the network $W_2 W_1$ using only the effective singular values $\hat{s}_{\alpha}(t)$ (this characterization was derived from the assumption of initially decoupled connectivity modes).
## Findings.
The form of the expression for $\hat{s}_{\alpha}(t)$ suggests that the effective singular values, or effective mode strengths, should follow a sigmoidal trajectory[^9]. These sigmoidal trajectories can be arbitrarily 'sharp' and represent rapid transitions from unlearned to learned. The 'sharpness' is controlled by the ratio $s_\alpha / \hat{s}_\alpha^0$: a larger target singular value relative to initialization produces a sharper transition. In a shallow linear network, by contrast, each mode's effective singular value grows exponentially rather than sigmoidally. The 'S-curve' (slow start, rapid transition, plateau) is a unique signature of depth, even without non-linearities.

<center>
<img src="Screenshot 2026-02-11 at 12.29.19 AM.png" width="400">
<figcaption>Figure 4: Dynamics of learning in a 2-layer neural network. Curves show the strength of the network’s representation of seven modes of the input-output correlation matrix over the course of learning. Red traces show analytical curves. Blue traces show simulation of full dynamics of a 2-layer linear network from small random initial conditions. Green traces show simulation of a nonlinear 2-layer network with tanh activation functions.</figcaption>
</center>

Some notable facts that we derived with theory:
- The time it takes for the network to learn a specific input-output mode is inversely proportional to that mode's singular value strength $s_\alpha$. We see this phenomenon in non-linear neural networks as well: they grasp the dominant, broad structure of the data (large $s$) very quickly, but take much longer to internalize the subtle details (small $s$).
- In deep linear networks, though the mapping is linear, the gradient descent dynamics are coupled and nonlinear. The derivation shows that the learning trajectory of a mode follows a sigmoid function. This proves that plateaus (periods of little apparent improvement) and sudden transitions of rapid learning are inherent to gradient descent in deep architectures, not just a side effect of activation functions like ReLU or Tanh.
## Conclusion.
We made a few assumptions to get to this point; the main ones being that the input data was whitened and that initially the connectivity modes were decoupled and aligned. These assumptions allowed us to simplify the problem from many intertwined matrix equations all the way down to decoupled scalar equations. Ultimately, all our assumptions are vindicated by the fact that the results we ended up with approximate the real thing well[^10] (see figure 4). Recall our decoupled and aligned connectivity modes assumption. na

Sources: [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/pdf/1312.6120), [A mathematical theory of semantic development in deep neural networks](https://arxiv.org/pdf/1810.10531)

[^1]: Here $\mu$ indexes the training examples ($\mu = 1, \dots, P$).

[^2]: A slight caveat: usually the hidden layer acts as a sort of bottleneck in the sense that $N_{2} < N_{1},N_{3}$. This means that the rank of $W_2 W_1$ will be constrained by the size of the hidden layer. In this case, the network will only be able to learn the top $N_{2}$ singular vectors of $\Sigma_{xy}$.

[^3]: Here $\alpha$ enumerates the modes ($\alpha = 1, \dots, r$, ordered so that $s_1 \geq s_2 \geq \cdots$).

[^4]: Observe that in figure 3, unlike the input modes, the output modes do not form a complete basis for the feature space. This is essentially because four categorical directions are enough to completely characterize the structure of the data.

[^5]: By rearranging, we can gain some intuition as to what these new matrices represent: $W_1 = \tilde{W}_1 V^\top, \space W_2 = U \tilde{W}_2$. The matrix $W_1$ takes inputs to the hidden layer. Since $V^\top$ ($=V^{-1}$) can be thought of as a change of basis from input space to the space of input modes, $\tilde{W}_1$ can be thought of as taking input modes to hidden neurons. Similarly, $W_2$ takes hidden neurons to outputs. Since $U$ can be thought of as a change of basis from the space of output modes to output space, $\tilde{W}_2$ can be thought of as taking hidden neurons to output modes. 

[^6]: Here, $\delta_{\alpha \beta}​=1$ if $\alpha=\beta$ and $\delta_{\alpha \beta}​=0$ if $\alpha \neq \beta$.

[^7]: A similar caveat to footnote 2: if $N_{2}<N_{1},N_{3}$, i.e. the hidden layer serves as a bottleneck, then we assume $\mathbf{a}_\alpha = a_{\alpha} \mathbf{r}_\alpha, \space \mathbf{b}_\alpha = b_{\alpha}\mathbf{r}_\alpha$ for $\alpha=\{1,\dots,N_{2}\}$ and $\mathbf{a}_\alpha=\mathbf{b}_\alpha=0$ for $\alpha=\{N_{2}+1,\dots\}$.

[^8]: Orthogonality of distinct connectivity modes is likely to be satisfied simply by the fact that [high dimensional random vectors are approximately orthogonal](https://math.stackexchange.com/questions/995623/why-are-randomly-drawn-vectors-nearly-perpendicular-in-high-dimensions).

[^9]: Recall that a sigmoid function has the form: $\frac{1}{1+e^{-x}}$.

[^10]: And they are of course also justified by the fact that they make the math easier and arguably more elegant.
