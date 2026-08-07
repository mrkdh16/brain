---
title: <% tp.file.title %>
draft: false
tags:
---
I learned so much new math going over this [tutorial](https://arxiv.org/pdf/2601.01010) on dynamical mean-field theory (DMFT). Here, I'll try my best to document what I learned.
# Linear Dynamical System with GOE Matrix
The setup is as follows:
$$
\begin{align}
\frac{d}{dt}h(t)=-Mh(t)+j(t)
\end{align}
$$
where $h,j\in \mathbb{R}^N$ and the matrix $M\in\mathbb{R}^{N\times N}$ is a symmetric Gaussian (GOE/Wigner) matrix: $M=\frac{1}{\sqrt{N}}A$ where $A=A^\top$ and $A_{ij}\sim\mathcal{N}(0,1)$. Perhaps in another note, we'll see how the same techniques used to analyze this system can be applied to a neural network. For now, with this setup, we'll learn and apply some fancy math (functional derivatives and Taylor expansions, the cavity method, Gaussian processes, Fourier transforms, and a little bit of random matrix theory) to informally derive [Wigner's semicircle law](https://en.wikipedia.org/wiki/Wigner_semicircle_distribution).
## Intuition for the system we're modeling
The way to think about the system that the above differential equation is modeling is as a time evolving network of sites $h$, interconnected and interacting according to the matrix $M$, under some external influence $j$ (hereon called the *source*). Without any external influence (i.e., $j(t)=0$), the sites undergo a somewhat boring evolution described in the next section.

<center>
<img src="Screenshot 2026-08-05 at 11.26.58 AM.png" width="300">
</center>

The value of a specific site $h_k(t)$ (at a specific time $t$) not only affects every other site as time moves forward, the signal propagates throughout the network and eventually returns to affect the value of $h_k(t')$ at some later time $t'>t$. The insight in the DMFT derivation below is that as the number of sites goes to infinity, $N\to\infty$, the distinct sites become *statistically equivalent* through central limit theorem (CLT)-style arguments.
## Power iteration
This is somewhat irrelevant to the actual DMFT, but it's a useful exercise to think about what happens when $j(t)=0$. When $j(t)=0$, the differential equation becomes $\frac{d}{dt}h(t)=-Mh(t)$. The solution is simply an exponential: 
$$
h(t)=e^{-Mt}h(0) = e^{-U\Sigma U^\top t}h(0) = Ue^{-\Sigma t}U^\top h(0)
$$
Redefining variables such that $\tilde{h}(t) = U^\top h(t)$, we see that
$$
(\tilde{h}(t))_{k} = e^{-\lambda_k t}(\tilde{h}(0))_k \implies \frac{\tilde h(t)}{|\tilde h(t)|}\to e_N \implies \frac{h(t)}{|h(t)|}\to u_{\min}
$$
where $u_{\min}$ is the eigenvector of $M$ corresponding to the minimum eigenvalue. This is essentially the continuous version of [power iteration](https://en.wikipedia.org/wiki/Power_iteration).
## Towards a self-consistent system of equations
We start by manipulating the original differential equation to get an integral. Start by multiplying $e^{Mt}$ to both sides of the equation and cleverly applying the product rule. 
$$
e^{Mt}\Big(\frac{d}{dt}h(t)+Mh(t)\Big)=e^{Mt}j(t)\implies \frac{d}{dt}\big[e^{Mt}h(t)\big]=e^{Mt}j(t).
$$
Applying the fundamental theorem of calculus, 
$$
e^{Mt}h(t)-h(0)=\int_0^t e^{Mt'}j(t')dt'.
$$
Finally, some algebra gets us to the form we want:
$$
h(t)=e^{-Mt}h(0)+\int_0^t e^{-M(t-t')}j(t')dt' = e^{-Mt}h(0)+\int_{0}^{\infty} e^{-M(t-t')}j(t')\Theta(t-t')dt'
$$
where $\Theta(t-t')=1$ if $t>t'$ and $0$ if $t \leq t'$ (Heaviside function). Using the Heaviside function allows us to set the upper limit of the integral to $\infty$ which will be useful later when applying a Fourier transform.
### The cavity method
The more general and canonical way of deriving the self-consistent system of equations is through the MSRDJ path integral. Here, we'll instead apply the cavity method, which leads to a more intuitive, albeit less formal, derivation. The following is a broad outline of how the cavity method will work:

1. Consider adding a new site $h_0(t)$ to the system, resulting in $N+1$ total sites.
2. Derive an expression for the time evolution of the new site $h_0(t)$ in the limit $N\to\infty$.
3. Argue that as $N\to\infty$, all sites become statistically equivalent such that the expression for $h_0(t)$ derived in step 2 is representative of how every site will evolve.

After we derive one expression for how every site will evolve, we'll leverage the time-translation-invariance (TTI) given by the linearity of the system to apply a Fourier transform and learn some statistical properties of our symmetric Gaussian random matrix $M$.

Let's start by seeing what happens when we add a new site $h_0(t) \in \mathbb{R}$ to the system:
$$
\frac{d}{dt}\begin{bmatrix}h_0(t) \\ \vert \\ \tilde h(t) \\ \vert\end{bmatrix}= 
-\frac{1}{\sqrt{N}}\left[ \begin{array}{c|ccc} a_{00} & a_{01} & \cdots & a_{0N} \\ \hline a_{01} & & & \\ \vdots & & A & \\ a_{0N} & & & \end{array} \right]
\begin{bmatrix}h_0(t) \\ \vert \\ \tilde h(t) \\ \vert\end{bmatrix}
+
\begin{bmatrix}j_0(t) \\ \vert \\ j(t) \\ \vert\end{bmatrix}
$$
where $a_0\in\mathbb{R}^N$ is the new random vector (the $i$th element is a standard Gaussian variable $a_{0i}\sim\mathcal{N}(0,1)$) connecting the new site $h_0(t)$ to the rest of the network (we're going to ignore $a_{00}$ for the most part, since it becomes irrelevant as $N$ gets very large). And $\tilde{h}(t)\in\mathbb{R}^N$ describes the trajectory of the original $N$ sites in this new system with an additional $N+1$th site in $h_0(t)$. Note that $\tilde{h}(t)$ will be different from $h(t)$ for any $t>0$ since the entire trajectory of $\tilde{h}(t)$ is affected by the new site $h_0(t)$. The above equation makes it clear that
$$
\frac{d}{dt}\tilde h(t) = -M\tilde h(t)+j(t) - \frac{1}{\sqrt N}h_0(t)a_0.
$$
We can thus think of $\tilde{h}(t)$ as the description of a system identical to $h(t)$ only with a shifted source: $j(t) \to j(t)-\frac{1}{\sqrt{N}}h_0(t)a_0$.

We can also derive an expression for $h_0(t)$:
$$
\frac{d}{dt}h_0(t) = -\frac{1}{\sqrt N}(a_{00}h_0(t)+a_0^\top\tilde h(t)+j_0(t)) \approx -\frac{1}{\sqrt N}a_0^\top\tilde h(t)+j_0(t)
$$
where we ignored the contribution of the $a_{00}h_0(t)$ term since its negligible compared to the contribution of $a_0^\top\tilde h(t)$, especially as $N\to\infty$. We want to substitute an expression for $\tilde{h}(t)$ such that everything on the right-hand-side is written in terms of the original sites $h(t)$. And we want the terms on the right-hand-side to concentrate (CLT-style) as $N\to\infty$. To this end, we're going to utilize what we found in the previous paragraph: that $\tilde{h}(t)$ describes a system identical to $h(t)$ only with a shifted source $j(t)-\frac{1}{\sqrt{N}}h_0(t)a_0$. 
#### Functional derivatives and Taylor expansions
At this point, we're going to have to introduce the first piece of heavy mathematical machinery: functional derivatives. 

A functional $F$ takes as input a function $j(t)$ and returns a scalar: $F[j] \in\mathbb{R}$. Crucially, the value of $F[j]$ depends on the function value of $j(t)$ for all inputs $t\in\mathbb{R}$, i.e., the entire history of $j$ contributes to the value of $F[j]$. Going one step further, we can imagine $F$ also being a function (a function which takes as input another function is called an operator) such that for any fixed $t\in\mathbb{R}$, $F[j](t)$ is a functional. Our vector of sites $h(t)$ is an example of an operator; it takes as input the source function $j(t)$ and outputs a new function: $h[j](t)$. This is most apparent in the integral formulation of the equation:
$$
h(t)= e^{-Mt}h(0)+\int_{0}^{\infty} e^{-M(t-t')}j(t')\Theta(t-t')dt'.
$$
Clearly the function $h(t)$ depends on the value of $j(t')$ for all $t'\in \mathbb{R}^+$ (technically the Heaviside function means that the dependence isn't this strict, but whatever). 

If $j$ is an input to $h$, can we differentiate with respect to it? Of course we can! The functional derivative $\frac{\delta h(t)}{\delta j(s)}$ is defined as the function that satisfies:
$$
\left[\frac{d}{d\varepsilon}h[j+\varepsilon v](t)\right]_{\varepsilon=0}=\int \frac{\delta h(t)}{\delta j(s)}v(s)ds
$$
where $v$ is some function in the same function space as $j$. Expanding the left-hand-side by the standard definition of a scalar derivative,
$$
\left[\frac{d}{d\varepsilon}h[j+\varepsilon v](t)\right]_{\varepsilon=0}=\lim_{\varepsilon\to0}\frac{F[j(s)+\varepsilon v(s)](t)-F[j(s)](t)}{\varepsilon}.
$$
We can let $v(s)=\delta(s-t')$ to get a more concrete expression for $\frac{\delta h(t)}{\delta j(t')}$:
$$
\int \frac{\delta h(t)}{\delta j(s)}\delta(s-t')ds=\frac{\delta h(t)}{\delta j(t')} = \lim_{\varepsilon\to0}\frac{h[j(s)+\varepsilon \delta(s-t')](t)-h[j(s)](t)}{\varepsilon}.
$$
Recall that $\displaystyle h_i(t)=(e^{-Mt}h(0))_i+\int_0^\infty\sum_l\big[e^{-M(t-s)}\big]_{il}j_l(s)\Theta(t-s)ds$. Plugging this into the definition of $\frac{\delta h(t)}{\delta j(t')}$,
$$
\frac{\delta h_i(t)}{\delta j_k(t')} = \lim_{\varepsilon\to0}\frac1\varepsilon\Big(\int_0^\infty \big[e^{-M(t-s)}\big]_{ik}\varepsilon\delta(s-t')\Theta(t-s)ds\Big) = \big[e^{-M(t-t')}\big]_{ik}\Theta(t-t').
$$
We're going to need this later...

Now equipped with the knowledge of functional derivatives, we're ready to use a functional Taylor expansion to approximate $\tilde{h}(t)$ as a perturbed version of $h(t)$. Recall that $\tilde{h}(t)$ is $h(t)$ with a perturbed input function $\tilde{j}(t)=j(t)-\frac{1}{\sqrt{N}}h_0(t)a_0$ where $-\frac{1}{\sqrt{N}}h_0(t)a_0$ is the perturbation. Since the perturbation $-\frac{1}{\sqrt{N}}h_0(t)a_0$ gets small as $N\to\infty$, we can approximate $\tilde{h}(t)$ by Taylor expanding $h(t)$ to first order:
$$
\tilde{h}(t)\approx h[j+\delta j](t) = h[j](t) + \int_{-\infty}^{t} \frac{\delta h(t)}{\delta j(t')^\top}\delta j(t')dt' + \text{higher order terms}.
$$
where $\delta j(t')=-\frac{1}{\sqrt{N}}h_0(t')a_0$ such that
$$
\tilde{h}(t)\approx h[j](t) -\frac{1}{\sqrt{N}}\int_{-\infty}^{t} \frac{\delta h(t)}{\delta j(t')^\top}h_0(t')a_0dt' + \text{higher order terms}.
$$
### Self-consistent equations
Plugging our first order Taylor approximation of $\tilde{h}(t)$ into our equation for $h_0(t)$,
$$
\frac{d}{dt}h_0(t)\approx \underbrace{-\frac{1}{\sqrt N}a_0^\top h(t)}_{\text{noise term}} + \underbrace{\frac{1}{N}\int_{-\infty}^{t}a_0^\top\frac{\delta h(t')}{\delta j(t')^\top}h_0(t')a_0dt'}_{\text{response term}} + j_0(t) + \text{higher order terms}.
$$
The important observation here is that $a_0$ is a random vector with $a_{0i}\sim\mathcal{N}(0,1)$, statistically independent of $h(t)$ (by construction). So, with $\frac{1}{\sqrt{N}}$ scaling, the noise term $u_0(t) := -\frac{1}{\sqrt N}a_0^\top h(t)$ converges to a Gaussian process as $N\to \infty$: $u_0(t)\sim\mathcal{GP}(0,C(t,t))$. This means that for any finite collection of times $\{t_1,\cdots,t_n\}$, the random vector $(u_0(t_1),\cdots,u_0(t_n))\in\mathbb{R}^n$ is jointly Gaussian with covariance matrix $\Sigma_{ij} = C(t_i,t_j)$, where 
$$
C(t,t') = \lim_{N\to\infty}\frac{1}{N}\sum_{k=1}^N h_k(t)h_k(t') = \lim_{N\to\infty}\frac{1}{N} h(t)^\top h(t')
$$ is the (deterministic as $N\to\infty$ by the law of large numbers) second-moment kernel of the hidden-unit features.