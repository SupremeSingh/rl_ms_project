# Learning compact features for a linear critic

*Part 3 of Building a Cheaper Critic · 14 October 2026*

Suppose a language model is halfway through solving a problem. Inside the network are vectors describing the question and the answer so far. A critic wants to use those vectors to predict eventual success. How much of that internal description does it need?

Keeping everything is tempting. More features might expose more useful relationships. But every extra coordinate makes a linear critic more expensive to fit, and some coordinates may describe details that do little to predict reward.

A useful alternative is to learn a small representation first, freeze it, and fit a linear value function on top. The representation can be nonlinear even though the final critic is linear in its weights. This article develops that idea and asks what sparsification adds to it.

## 1. Start with the state, not the completed answer

For a language model, a state is the question together with the tokens already generated. Consider:

> There are twelve apples. Three people share them equally. How many does each receive?

After the prefix “We divide twelve by,” the critic estimates expected reward if the policy continues. It must not read the future token “three” or the final answer when constructing this state's features.

At each prefix, a transformer supplies a hidden vector from each block. We can read one layer, or concatenate several:

\[
x(s)=[h_{\ell_1}(s);h_{\ell_2}(s);\ldots;h_{\ell_K}(s)]\in\mathbb R^D.
\]

All these vectors come from the same last available token position. They describe different stages of processing the same prefix, not different future steps. Normalizing each layer can prevent its magnitude from dominating merely because its activations have a larger scale.

Concatenation supplies more coordinates; it does not guarantee more useful information. A learned compressor tries to keep the distinctions that matter for the prediction task.

## 2. Compression needs a purpose

Principal component analysis, or PCA, retains directions with large variance. If a cloud of points spreads mostly along two axes, PCA can represent it with those axes while minimizing linear reconstruction error.

Predicting reward asks a different question. Imagine one direction records answer length and varies enormously. Another, smaller direction distinguishes valid from invalid arithmetic. A reconstruction method can prefer the first even when a critic needs the second.

This does not make PCA a bad method. It makes its objective different. A supervised compressor instead asks: which small set of features helps predict the eventual outcome?

Let \(y\in\{0,1\}\) indicate whether one sampled continuation succeeds. The label is one observation of the return, not the exact value of the prefix. Different continuations from the same prefix can have different outcomes. Learning across examples estimates the relationship between the prefix and those outcomes.

## 3. A small nonlinear bottleneck

First center and scale the input using statistics from training data. Write that processed vector as \(\widetilde x\). A simple bottleneck is

\[
z(s)=\operatorname{ReLU}(\widehat W\widetilde x(s)+b),
\qquad \widehat W\in\mathbb R^{m\times D},\quad m<D.
\]

ReLU acts coordinate by coordinate: \(\operatorname{ReLU}(u)=\max(0,u)\). The output has only \(m\) coordinates, so it is compressed by construction.

For an illustrative input \(\widetilde x=(2,1,-1)^\top\), take

\[
\widehat W=
\begin{bmatrix}
1/\sqrt2&-1/\sqrt2&0\\
0&0&1
\end{bmatrix},\qquad b=0.
\]

Then \(z=(1/\sqrt2,0)^\top\). Three input coordinates become two learned combinations, one of which is inactive. The numbers are a toy example, not measurements of a language model.

The rows can be constrained to unit norm so that their scale is controlled. Unit norm is not orthogonality: two unit-length rows can point in the same direction. Nor does the matrix have to be sparse. Dense weights can produce sparse activations.

## 4. Teach the bottleneck what to preserve

Attach a temporary predictor:

\[
\widehat p(s)=\sigma(a^\top z(s)+c),
\qquad \sigma(u)=\frac{1}{1+e^{-u}}.
\]

Train it with binary cross-entropy:

\[
\mathcal L_{\mathrm{prediction}}
=-\frac1B\sum_{i=1}^{B}
\left[y_i\log\widehat p_i+(1-y_i)\log(1-\widehat p_i)\right].
\]

The loss updates both the bottleneck and its temporary readout. It encourages the bottleneck to expose information a simple predictor can use. It does not require reconstructing every aspect of the original vector.

Why introduce a nonlinear layer if the destination is a linear critic? Because a linear decision in the new coordinates need not be linear in the old ones. ReLU changes which features are active in different regions of the input space. A final weighted sum can therefore express a piecewise-linear function of the original representation.

## 5. Sparsity is a second, separate preference

A narrow vector can still have every coordinate active. To encourage a more selective code, add an activation penalty:

\[
\mathcal L=
\mathcal L_{\mathrm{prediction}}
+\beta\frac{1}{Bm}\sum_{i=1}^{B}\sum_{j=1}^{m}|z_{ij}|.
\]

Since ReLU outputs are nonnegative, the penalty is simply \(\beta\operatorname{mean}(z)\). It asks the network to predict well while using less total activation. ReLU provides exact zeros; the penalty encourages smaller activations and can encourage more inactive units. It does not fix the number of zeros.

There are two knobs with different jobs. The width \(m\) limits how many coordinates exist. The coefficient \(\beta\) controls how costly it is to activate them. Turning up \(\beta\) too far can destroy useful information or leave units inactive on every input.

Scale also matters. Without constraints, a network can shrink its features and enlarge the readout weights while leaving predictions unchanged. Normalizing projection rows and regularizing readout weights discourages this shortcut, but does not guarantee a unique or meaningful sparse code.

A decisive control is the same bottleneck with \(\beta=0\). Both models compress and use ReLU; only one explicitly penalizes activation. Comparing them separates the benefit of learned compression from the additional benefit of sparsity.

## 6. The connection to geometric reinforcement learning

Tiwari, Gottesman and Konidaris study how neural policy learning can induce locally low-dimensional structure in continuous-control state spaces. Their theoretical result depends on assumptions about the policy, dynamics and training regime. It is a motivation for investigating useful low-dimensional structure, not a dimensionality theorem for transformer activations. [Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces](https://cs.brown.edu/people/gdk/pubs/geometry_neural_rl.pdf)

In §5.4, their practical intervention uses a CRATE-inspired sparsification layer. Equation 12 is

\[
z^{+}=\operatorname{ReLU}\left(z+\alpha W^\top(z-Wz)-\alpha\lambda\mathbf1\right).
\]

This is a structured residual transformation with a correction term and threshold. Its square matrix preserves the layer's width. The paper places it inside policy and value networks for SAC; it does not prescribe a smaller supervised bottleneck for an LLM critic. [Paper, §5.4 and Appendix M](https://cs.brown.edu/people/gdk/pubs/geometry_neural_rl.pdf)

The bottleneck above borrows the motivation to seek selective representations, but uses a different mechanism: supervised dimensional reduction plus an activation penalty. It neither implements the paper's residual layer nor explicitly optimizes rate reduction or enforces orthogonality. Calling it a sparse orthogonal projection would therefore be inaccurate.

## 7. Freeze the representation, then fit the value

Once the bottleneck is trained, freeze its parameters and preprocessing. Discard the temporary sigmoid predictor. Define

\[
\phi(s)=[z(s);1],\qquad \widehat V(s)=w^\top\phi(s).
\]

Now ridge or LSTD can fit \(w\). As [Article 2](02-a-critic-built-from-matrices.md) explains, LSTD requires linearity in the fitted weights, not linearity in the original text. A frozen nonlinear feature map is compatible with that requirement.

For LSTD, features must be available at consecutive states. Random prefixes are useful for training and evaluating the bottleneck, but are not automatically adjacent transitions. At true termination, use a zero next-state feature, including the intercept.

Freezing is essential when accumulating matrices. If the bottleneck changes, the same coordinate may mean something different. Old sufficient statistics generally cannot be reused unchanged under a new nonlinear representation.

## 8. What would make this useful?

A smaller critic needs roughly quadratic matrix storage and a dense solve with cubic scaling in its feature dimension. Reducing width can therefore matter substantially. But zero activations do not automatically accelerate dense software, and extracting several transformer layers still has a cost.

A fair test compares the learned representation with PCA at equal width, the original concatenation, and a strong single-layer representation. Keep data and fitting budgets aligned. Measure probability error and calibration as well as classification accuracy. Measure feature extraction, compression and fitting costs together.

Finally, supervised features can leak labels just as a fitted head can. Split by question rather than scattering prefixes of one answer across training and evaluation. When using cross-fitting for policy updates, protect the representation-training stage as well as the final critic fit.

In the future, could a small, selectively active representation preserve the information needed for good decisions while discarding much of what a language model needs only to produce fluent text?
