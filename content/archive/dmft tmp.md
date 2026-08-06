---
title: <% tp.file.title %>
draft: true
tags:
---
Here's the full transcription I was able to pull from the handwritten pages (50–55). I've kept your original grouping/order. A few spots on p.51's side-scratch work were ambiguous in the handwriting — I've marked those with a note.

## Page 50

$$\frac{d}{dt}h(t) = -Mh(t) + j(t) \quad \text{where } M=M^T,\ M=\tfrac{1}{\sqrt N}A,\ A_{ij}\sim N(0,1)$$

$$\text{If } j(t)=0,\text{ then } \frac{d}{dt}h(t)=-Mh(t) \text{ is trivial: } h(t)=e^{-Mt}h(0) = e^{-U\Sigma U^Tt}h(0) = Ue^{-\Sigma t}U^Th(0)$$

$$\Rightarrow \tilde h(t)=Uh(t),\quad (\tilde h(t))_i = e^{-\lambda_i t}(\tilde h(0))_i$$

$$\Rightarrow \frac{\tilde h(t)}{|\tilde h(t)|}\to e_d \ \ (h,\tilde h\in\mathbb R^d)\ \text{ s.t. } \frac{h(t)}{|h(t)|}\to u_{\min}$$

where $u_{\min}$ is the eigenvector of $M$ corresponding to the smallest eigenvalue.

$$\frac{d}{dt}h(t)+Mh(t)=j(t)$$ $$e^{Mt}\Big(\frac{d}{dt}h(t)+Mh(t)\Big)=e^{Mt}j(t)$$ $$\frac{d}{dt}\big[e^{Mt}h(t)\big]=e^{Mt}j(t)$$ $$e^{Mt}h(t)-h(0)=\int_0^t e^{Mt'}j(t'),dt'$$ $$h(t)=e^{-Mt}h(0)+\int_0^t e^{-M(t-t')}j(t'),dt' = e^{-Mt}h(0)+\int_{-\infty}^{\infty} e^{-M(t-t')}j(t'),\Theta(t-t'),dt'$$

**Cavity Method.**



① add a new site $h_0(t)$

$$\vec h(t)=\begin{bmatrix}h_1(t)\ \vdots \ h_N(t)\end{bmatrix} \longrightarrow \vec h_{\text{new}}(t)=\begin{bmatrix}h_0(t)\ \tilde h(t)\end{bmatrix}$$

Adding a new site at the "start" changes the dynamics of the whole process $(h(t)\to \tilde h(t))$.

$$\frac{d}{dt}\vec h_{\text{new}}(t) = -M_{\text{new}}\vec h_{\text{new}}(t)+\vec j(t) = -\begin{bmatrix}M & a\ a_0^T & 0\end{bmatrix}\begin{bmatrix}\tilde h(t)\ h_0(t)\end{bmatrix}+\begin{bmatrix}\vec j(t)\ j_0(t)\end{bmatrix}$$

$$\Rightarrow \frac{d}{dt}\tilde h(t) = -M\tilde h(t)+j(t) - \frac{1}{\sqrt N}h_0(t)a_0$$

Shifted source: $j(t)\to j(t)-\tfrac{1}{\sqrt N}h_0(t)a_0$, where $(a_0)_i\sim N(0,1)$.

Taylor expand $h(t)$, centered at $j(t)$, w/ perturbation $\delta j(t')=-\tfrac{1}{\sqrt N}h_0(t')a_0$:

$$h[j+\delta j](https://claude.ai/chat/t) = h[j](https://claude.ai/chat/t) + \int_{-\infty}^{t} dt',\frac{\delta h(t)}{\delta j(t')^T}\delta j(t') + O(N^{-1})$$

$$\Rightarrow \tilde h(t) = h(t) - \frac{1}{\sqrt N}\int_{-\infty}^{t}dt',\frac{\delta h(t)}{\delta j(t')^T}h_0(t')a_0 + O(N^{-1})$$

And $$\frac{d}{dt}h_0(t)\approx -\frac{1}{\sqrt N}a_0\cdot\tilde h(t)+j_0(t)$$ $$= \underbrace{-\frac{1}{\sqrt N}a_0\cdot h(t)}_{:=v_0(t)\ (\text{noise term})} + \underbrace{\frac{1}{N}\int_{-\infty}^{t}dt',a_0^T\frac{\delta h(t')}{\delta j(t')^T}h(t'),a_0}_{\text{response term}} + j_0(t) + O(N^{-3/2})$$

As $N\to\infty$, $a_0$ Gaussian w/ $a_0\perp h(t) \xrightarrow{\text{CLT}} v_0(t)\sim GP(0,C(t,t'))$ where $C(t,t')=\frac1N\sum_{i=1}^N h_i(t)h_i(t')$

$$\frac1N\int_{-\infty}^t dt',a_0^T\frac{\delta h(t')}{\delta j(t')^T}h(t')a_0 \xrightarrow{\text{LLN}} \int_{-\infty}^t dt'\ \underbrace{\Big[\frac1N\sum_{i=1}^N\frac{\delta h_i(t')}{\delta j_i(t')}\Big]}_{:=R(t,t')} h(t') + O(N^{-1/2})$$

## Page 51

$$y=f(x_1,\dots,x_n) \qquad x_k\to x_k+\varepsilon_k$$

$$f(x_1+\varepsilon_1,\dots,x_n+\varepsilon_n)=f(x)+\sum_{k=1}^n\frac{\partial f}{\partial x_k}\varepsilon_k+\frac{1}{2!}\sum_{k,l=1}^n\frac{\partial^2 f}{\partial x_k\partial x_l}\varepsilon_k\varepsilon_l+\cdots$$

As $n\to\infty$, $\begin{bmatrix}x_1\ \vdots\ x_n\end{bmatrix}\to j(t)$ (discrete index $k\to$ cts index $t$)

$$F[j+\delta j]=F[j]+\int dt,\frac{\delta F}{\delta j(t)}\delta j(t) + \frac{1}{2!}\int dt,dt',\frac{\delta F}{\delta j(t)\delta j(t')}\delta j(t)\delta j(t')+\cdots$$

$$\left{\ \lim_{\varepsilon\to0}\frac{F[j(s)-\varepsilon\delta(s-t)]-F[j(s)]}{\varepsilon}\ \right}$$

$$DF[j](https://claude.ai/chat/t;v)=\frac{d}{d\varepsilon}\Big|_{\varepsilon=0}F[j+\varepsilon v](https://claude.ai/chat/t)=\lim_{\varepsilon\to0}\frac{F[j+\varepsilon v](https://claude.ai/chat/t)-F[j](https://claude.ai/chat/t)}{\varepsilon}=\int dt',K(t,t')v(t'),\qquad K(t,t'):=\frac{\delta F(t)}{\delta j(t')}$$

$$\frac{d}{d\varepsilon}\Big|_{\varepsilon=0}F[j+\varepsilon v]=\int dt',\frac{\delta F}{\delta j(t')}v(t')$$

Let $v(s)=\delta(s-t)$ to recover $\displaystyle\int dt',\frac{\delta F}{\delta j(t')}\delta(t'-t)=\frac{\delta F}{\delta j(t')}$

$$h_i[j+\delta j](https://claude.ai/chat/t)=h_i[j](https://claude.ai/chat/t)+\int dt',\frac{\delta h_i(t)}{\delta j(t')^T}\delta j(t') + O(N^{-1})$$

$$\frac{\delta h_i(t)}{\delta j_k(t')} = \lim_{\varepsilon\to0}\frac{h_i[j_k(s)+\varepsilon\delta(s-t')](https://claude.ai/chat/t)-h_i[j_k(s)]}{\varepsilon}$$

$$h_i(t)=\cdots+\int_0^t\sum_l\big[e^{-M(t-s)}\big]_{il}j_l(s),ds$$

$$\lim_{\varepsilon\to0}\frac1\varepsilon\sum_l\int_0^t ds,\big[e^{-M(t-s)}\big]_{il},\varepsilon,\delta(t-t') = \sum_l\big[e^{-M(t-t')}\big]_{il}$$

_(Side scratch work on this page, not fully certain of intent/order:)_

$$e^{+Mt}h_0(t)a_0 \qquad\qquad \int_{-\infty}^{t}e^{-M(t-t')}h_0(t')a_0,dt'$$

$$e^Ae^B=\Big(1+A+\tfrac{A^2}{2!}+\cdots\Big)\Big(1+B+\tfrac{B^2}{2!}+\cdots\Big)$$ $$Ue^{\Sigma t}U^T = \Big(\sum_{n\ge0}\frac{A^n}{n!}\Big)\Big(\sum_n\frac{B^n}{n!}\Big) = \sum_{i,j\ge0}\frac{A^iB^j}{i!,j!}$$

$$h(t)=\int_{-\infty}^{\infty}e^{-M(t-t')}j(t'),\Theta(t-t'),dt'+\cdots$$ $$\Rightarrow \frac{\delta h_i(t)}{\delta j_k(t')} = \big[e^{-M(t-t')}\big]_{ik}\Theta(t-t')$$

_(A few more isolated fragments lower on the page, likely unrelated asides — a feedforward-net formula $f(x)=\sum_i v_i,\phi(w_i^Tx+b_i)$, and a sum $\sum_{i=1}^N a_{0,i}h_i(t)$ with $a_{0,i}\sim N(0,1)$ — these were too faint/cropped to transcribe with confidence.)_

## Page 52

$$\frac{\delta F(t)}{\delta j(t')} \text{ is defined as the function that satisfies } \frac{d}{d\varepsilon}\Big|_{\varepsilon=0}F[j+\varepsilon v](https://claude.ai/chat/t)=\int dt',\frac{\delta F(t)}{\delta j(t')}v(t')$$

where LHS $=\displaystyle\lim_{\varepsilon\to0}\frac{F[j+\varepsilon v](https://claude.ai/chat/t)-F[j](https://claude.ai/chat/t)}{\varepsilon}$

$$\Rightarrow \text{if } v(s)=\delta(s-t'),\text{ then } \frac{\delta F(t)}{\delta j(t')}=\lim_{\varepsilon\to0}\frac{F[j(s)+\varepsilon\delta(s-t')]-F[j(s)]}{\varepsilon}$$

Since $\displaystyle h_i(t)=\cdots+\int_0^\infty ds\sum_l\big[e^{-M(t-s)}\big]_{il},j_l(s),\Theta(t-s)$,

$$\frac{\delta h_i(t)}{\delta j_k(t')} = \lim_{\varepsilon\to0}\frac1\varepsilon\Big(\int_0^\infty ds,\big[e^{-M(t-s)}\big]_{ik},\varepsilon,\delta(s-t'),\Theta(t-s)\Big)$$ $$= \int_0^\infty ds,\big[e^{-M(t-s)}\big]_{ik},\delta(s-t'),\Theta(t-s) = \big[e^{-M(t-t')}\big]_{ik},\Theta(t-t')$$

A process $u(t)$ is a GP if $\forall$ finite collection of times ${t_1,\dots,t_n}$, the random vector $(u(t_1),\dots,u(t_n))\in\mathbb R^n$ is jointly Gaussian.

## Page 53

As $N\to\infty$, all sites $h_i(t)$ become statistically equivalent.

$$\frac{d}{dt}h(t)=u(t)+\int_{-\infty}^t dt',R(t,t')h(t')+j(t),\qquad u(t)\sim GP(0,C(t,t')),\qquad R(t,t')=\frac1N\sum_{i=1}^N\frac{\delta h_i(t)}{\delta j_i(t')}$$

$$\frac{d}{dt}\frac{\delta h(t)}{\delta u(t')} = \delta(t-t') + \int_{-\infty}^t dt'',R(t,t'')\frac{\delta h(t'')}{\delta u(t')}$$

Assume $\displaystyle\frac1N\sum_{i=1}^N\frac{\delta h_i(t)}{\delta j_i(t')}\to \frac{\delta h(t)}{\delta j(t')}$ (self-averaging as $N\to\infty$)

$$\frac{d}{dt}R(t,t') = \delta(t-t') + \int_{-\infty}^t dt'',R(t,t'')R(t'',t')$$

_(side substitution notes: $\tau=t-t',\ s=t''-t'$, etc.)_

With $\tau=t-t'$: $$\frac{d}{d\tau}R(\tau) = \delta(\tau) + \int_{-\infty}^\infty d\sigma,R(\tau-\sigma)R(\sigma)$$

$$\frac{\delta h_i(t)}{\delta u_k(t')}=\big[e^{-M(t-t')}\big]_{ik}\Theta(t-t') = \big[e^{-M\tau}\big]_{ik}\Theta(\tau)$$

**Fourier Transform**

$$R(\omega)=\int_{-\infty}^\infty d\tau,e^{i\omega\tau}R(\tau),\qquad R(\tau)=\frac{1}{2\pi}\int d\omega,e^{-i\omega\tau}R(\omega)$$

$$\frac{d}{d\tau}R(\tau) = \frac{1}{2\pi}\int d\omega,e^{-i\omega\tau}(i\omega R(\omega)) \ \longrightarrow\ \widehat{\tfrac{d}{d\tau}R(\tau)} = i\omega R(\omega)$$

$$\int d\tau,e^{i\omega\tau}\Big[\int_{-\infty}^\infty ds,R(\tau-s)R(s)\Big] = \int ds,R(s)\int d\tau,e^{i\omega\tau}R(\tau-s) = \int ds,e^{-i\omega s}R(s)\int d\tau,e^{-i\omega(\tau-s)}R(\tau-s)$$ $$= \int ds,e^{-i\omega s}R(s)\int du,e^{-i\omega u}R(u) = R(\omega)^2$$

$$\therefore\ i\omega R(\omega) = 1+R(\omega)^2$$ $$R(\omega)^2 - i\omega R(\omega)+1=0 \ \Rightarrow\ R(\omega)=\frac{i\omega\pm\sqrt{(i\omega)^2-4}}{2}$$

$$R(\tau)=\frac1N\sum_{i=1}^N\frac{\delta h_i}{\delta j_i}(\tau) = \frac1N\sum_{i=1}^N\big[e^{-M\tau}\big]_{ii}\Theta(\tau) = \frac1N,\mathrm{Tr},e^{-M\tau},\Theta(\tau) = \frac1N\sum_{i=1}^N e^{-\lambda_i\tau}\Theta(\tau) = \int d\lambda\underbrace{\Big[\frac1N\sum_{i=1}^N\delta(\lambda-\lambda_i)\Big]}_{:=\rho(\lambda)\ (\text{spectral density})}e^{-\lambda\tau}\Theta(\tau)$$

$$R(\omega)=\int d\tau,e^{-i\omega\tau}R(\tau) = \int d\tau,e^{-i\omega\tau}\Big[\frac1N,\mathrm{Tr},e^{-M\tau}\Big]\Theta(\tau) = \frac1N\int d\tau,e^{-(i\omega+M)\tau}\Theta(\tau)$$ $$=\int d\tau,e^{-i\omega\tau}\int d\lambda,\rho(\lambda)e^{-\lambda\tau}\Theta(\tau) = \int d\lambda,\rho(\lambda)\int d\tau,e^{-(i\omega+\lambda)\tau}\Theta(\tau) = \int d\lambda,\frac{\rho(\lambda)}{i\omega+\lambda}$$ $$= \frac1N,\mathrm{Tr}\big[i\omega+M\big]^{-1} = \int d\lambda,\frac{\rho(\lambda)}{i\omega+\lambda}$$

## Page 54

$$f\mapsto f*g$$ $$(e^{i\omega\cdot}*g)(\tau)=\int ds,e^{i\omega(\tau-s)}g(s)=e^{i\omega\tau}\underbrace{\int ds,e^{-i\omega s}g(s)}_{:=G(\omega)} = G(\omega)\cdot e^{i\omega\tau}$$

$$f_\mu = \frac{1}{\gamma\sqrt N}w^L\cdot\phi(h_\mu^L),\qquad h_\mu^{\ell+1}=\frac{1}{\sqrt N}W^\ell\phi(h_\mu^\ell),\qquad h_\mu^1=\frac{1}{\sqrt D}W^0x_\mu$$

$$W_{ij}^\ell\sim N(0,1),\qquad \dot\theta=-\gamma^2\nabla_\theta L$$ $$\dot L=\nabla_\theta L\cdot\dot\theta=-\gamma^2(\nabla_\theta L)^2$$

$$g_\mu^\ell = \gamma\sqrt N,\frac{\partial f_\mu}{\partial h_\mu^\ell} = \dot\phi(h_\mu^\ell)\odot z_\mu^\ell,\qquad z_\mu^\ell = \frac{1}{\sqrt N}W^{\ell T}g_\mu^{\ell+1}$$

$$\Theta_{\mu\alpha}^\ell(t,s) = \frac1N\phi(h_\mu^\ell(t))\phi(h_\alpha^\ell(s)),\qquad G_{\mu\alpha}^\ell(t,s)=\frac1N g_\mu^\ell(t)g_\alpha^\ell(s)$$

$${\Theta^\ell,G^\ell}_{\ell=1}^L \to K_{\mu\alpha}^{NTK}(t,s) = \nabla_\theta f_\mu(t)^T\nabla_\theta f_\alpha(s) = \frac{1}{\gamma^2}\sum_{\ell=0}^L G_{\mu\alpha}^{\ell+1}(t,s),\Theta_{\mu\alpha}^\ell(t,s)$$

$$\frac{d}{dt}f_\mu(t) = \nabla_\theta f_\mu(t)\cdot(-\gamma^2\nabla_\theta L) = -\gamma^2\nabla_\theta f_\mu(t)\sum_{\alpha=1}^P\nabla_\theta f_\alpha(t),\frac{\partial L}{\partial f_\alpha}\Big|_{f(t)}$$ $$= \sum_{\alpha=1}^P K_{\mu\alpha}^{NTK}(t,t),\Delta_\alpha(t),\qquad \Delta_\alpha(t) = -\frac{\partial L}{\partial f_\alpha}\Big|_{f_\alpha(t)}$$

w/ base cases $G_{\mu\alpha}^{L+1}(t,s)=1$, $\Theta_{\mu\alpha}^0(t,s)=K_{\mu\alpha}^x=\frac1D x_\mu^Tx_\alpha$

mean-field-limit: $N,\gamma\to\infty,\ \gamma_0=\dfrac{\gamma}{\sqrt N}=O_N(1)$

## Page 55

$$\frac{\partial f_\mu}{\partial h_\mu^\ell} = \frac{\partial f_\mu}{\partial h_\mu^L}\cdots\frac{\partial h_\mu^{\ell+1}}{\partial h_\mu^\ell} = \frac{1}{\gamma\sqrt N}w^L\cdot\dot\phi(h_\mu^L)\cdots\frac{1}{\sqrt N}W^\ell\dot\phi(h_\mu^\ell) = \frac1\gamma,g_\mu^{\ell+1}\cdot\frac{1}{\sqrt N}W^\ell\dot\phi(h_\mu^\ell)$$

$$\nabla_\theta f_\mu(t) = \begin{bmatrix}\nabla_{W^0}f_\mu(t)\ \vdots\ \nabla_{W^L}f_\mu(t)\end{bmatrix}$$

$$\nabla_{W^\ell}f_\mu(t) = \frac{\partial f_\mu(t)}{\partial h_\mu^{\ell+1}}\cdot\frac{\partial h_\mu^{\ell+1}}{\partial W^\ell} = \frac{1}{\gamma\sqrt N}g_\mu^{\ell+1}\cdot\frac{1}{\sqrt N}\phi(h_\mu^\ell)$$

$$\nabla_\theta L = \sum_{\alpha=1}^P \nabla_\theta f_\alpha,\frac{\partial L}{\partial f_\alpha}\Big|_{f(t)}$$

$${u_\mu^\ell(t)}_{\mu\in[P],t\in\mathbb R_+}\sim GP(0,\Theta^{\ell+1}),\qquad {r_\mu^\ell(t)}_{\mu\in[P],t\in\mathbb R_+}\sim GP(0,G^{\ell+1})$$

$$h_\mu^\ell(t) = u_\mu^\ell(t) + \gamma_0\int_0^t ds\sum_{\alpha=1}^P\big[A_{\mu\alpha}^{\ell-1}(t,s)+\Delta_\alpha(s)\Theta_{\mu\alpha}^{\ell-1}(t,s)\big]z_\alpha^\ell(s)\dot\phi(h_\alpha^\ell(s))$$

$$z_\mu^\ell(t) = r_\mu^\ell(t) + \gamma_0\int_0^t ds\sum_{\alpha=1}^P\big[B_{\mu\alpha}^\ell(t,s)+\Delta_\alpha(s)G_{\mu\alpha}^{\ell+1}(t,s)\big]\phi(h_\alpha^\ell(s))$$

$$\gamma_0=0 \Rightarrow h_\mu^\ell(t)=u_\mu^\ell(t)\sim GP(0,\Theta^{\ell-1}) = GP\Big(0,\ \tfrac1N\phi(h_\mu^{\ell-1}(t))\phi(h_\mu^{\ell-1}(s))\Big)$$

$$h_\mu^1(t)\sim GP(0,\Theta^0)\ \text{where}\ \Theta_{\mu\alpha}^0(t,s) = \frac1D x_\mu^Tx_\alpha$$

$$h_\mu^2(t)\sim GP(0,\Theta^1)\ \text{where}\ \Theta_{\mu\alpha}^1(t,s) = \frac1N\phi(h_\mu^1(t))\phi(h_\alpha^1(s))$$

$$f(x;\theta)=0\ \text{at init (centered)}$$ $$\tilde f(x;\theta)=\alpha f(x;\theta) \quad\rightsquigarrow\quad \alpha\to\infty \Rightarrow \text{lazy}$$ $$\alpha=\frac{\gamma}{1/(\delta\sqrt N)},\quad \eta\leftarrow \eta N \Rightarrow \text{rich (}\mu\text{P)}$$

(side note: feature learning $\perp N$)

$$\frac{1}{\gamma\sqrt N}f(x;\theta),\qquad \eta N\ \text{lr}$$

A couple of notes: on page 51 there's a cluster of disconnected side-scratch computations (matrix exponential product checks, a feedforward net formula) that were faint/ambiguous in the handwriting — I flagged those inline rather than guess. If you want me to take another pass at zooming into any specific spot, let me know which page/region.