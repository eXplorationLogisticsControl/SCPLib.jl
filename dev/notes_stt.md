# State-transition tensor model

Consider a state $x \in \mathbb{R}^{n_x}$. 
Derivatives of the STT are defined via

$$
\begin{aligned}
\dot{\Phi}(t) &= A(t) \Phi(t,t_0)
\\
\dot{\Psi}(t) &= \Phi(t,t_0) \cdot \dfrac{\partial^2 f}{\partial x^2} \Phi(t,t_0) + A(t) \Psi(t,t_0)
% Φ[j, r] * params.d2f_dy2[p, i, j] * Φ[i, q] + params.df_dy[p, i] * Ψ[i, q, r]
\end{aligned}
$$

where $\Phi \in \mathbb{R}^{n_x \times n_x}$ and $\Psi \in \mathbb{R}^{n_x \times n_x \times n_x}$.
Hence, we need to propagate $n_x + n_x^2 + n_x^3$ state variables (assuming we do not leverage the symmetry of $\Psi$).
For a $n_x = 6$ DOF system, that would be $6 + 36 + 216 = 258$ variables.