---
title: Energy
draft: true
tags:
---
One of the first things one learns in an introductory physics class is that the mechanical energy of a system does not change as long as there is no external force. We also learn that for some special forces such as the gravitational force, even if the force acts on a system, the total mechanical energy stays the same. A gravitational force acting on an apple in free fall may decrease its potential energy, but the mechanical energy will stay constant as the kinetic energy will increase.

With the help of calculus, we can quantify this by saying that some forces are *conservative*, i.e. there exists some function $U$ such that $F = -\nabla U$, or in the one-dimensional case,
$$
F=-\frac{dU}{dx}.
$$
We define $U(x)$ as the *potential energy*. Using the Fundamental Theorem of Calculus, we show that 
$$
\int_{}^{} Fdx = -\Delta U.
$$
With Newton's Second Law and some clever manipulation we can also show that
$$
\int_{}^{} Fdx = \int m \frac{dv}{dt}dx = \int m \frac{dv}{dt}vdt= \int \frac{m}{2} \frac{d}{dt}\big[v^2\big]dt=\Delta\frac{1}{2}mv^2 = \Delta KE,
$$
Where we defined the quantity $\frac{1}{2}mv^2$ as the *kinetic energy* $KE$. We see that as long as $F$ is the only force acting on a system,
$$
\Delta KE + \Delta U = \int Fdx - \int Fdx = const.
$$
We define the sum of the kinetic energy and potential energy as the *mechanical energy* which is conserved as long as the forces acting on a system are conservative.

It can be helpful to plot the total mechanical energy $E$ and potential energy $U$ as functions of position. The reason is that the relation $F = -dU/dx$ allows us to think of the system in question simply as a ball rolling down a hill. 

