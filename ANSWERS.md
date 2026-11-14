# Assignment 4 – Theory Answers

## Part 2: Building Blocks of a Diffusion Model

### Question 1 – Sinusoidal timestep embedding

The scalar timestep `t` is turned into a `d`-dimensional vector by evaluating sines and cosines of `t` at `d/2` geometrically spaced frequencies (max period `P = 10000`):

$$
\omega_i = P^{-\,i/(d/2)} = \exp\!\Big(-\frac{\ln P \cdot i}{d/2}\Big), \qquad i = 0,1,\dots,\tfrac{d}{2}-1
$$

$$
\text{emb}(t)_i =
\begin{cases}
\sin(t\,\omega_i), & 0 \le i < d/2 \\[2pt]
\cos(t\,\omega_{\,i-d/2}), & d/2 \le i < d
\end{cases}
$$

So the first half of the vector holds sines and the second half the cosines of the same frequencies. (The original Transformer interleaves them instead: $PE_{2i}=\sin(t/P^{2i/d})$, $PE_{2i+1}=\cos(t/P^{2i/d})$ – same frequencies, different ordering.)

Steps: (1) compute the `d/2` frequencies, (2) multiply each by `t`, (3) take `sin` and `cos`, (4) concatenate. Low indices oscillate fast (fine resolution between neighbouring `t`), high indices oscillate slowly (coarse position). The embedding is then usually passed through a small MLP and added to the feature maps of each UNet block. Implemented in `helper_lib/model.py::SinusoidalEmbedding`.

### Question 2 – d = 8, t = 1, P = 10000

`d/2 = 4` frequencies: $\omega_i = 10000^{-i/4}$

| i | $\omega_i$ | $\sin(1\cdot\omega_i)$ | $\cos(1\cdot\omega_i)$ |
|---|---|---|---|
| 0 | $10000^{0} = 1$ | sin(1) = 0.841471 | cos(1) = 0.540302 |
| 1 | $10000^{-1/4} = 0.1$ | sin(0.1) = 0.099833 | cos(0.1) = 0.995004 |
| 2 | $10000^{-2/4} = 0.01$ | sin(0.01) = 0.010000 | cos(0.01) = 0.999950 |
| 3 | $10000^{-3/4} = 0.001$ | sin(0.001) = 0.001000 | cos(0.001) = 0.9999995 |

Embedding (sin half, then cos half):

$$
\text{emb}(1) = [0.8415,\ 0.0998,\ 0.0100,\ 0.0010,\ 0.5403,\ 0.9950,\ 0.99995,\ 1.0000]
$$

(Interleaved Transformer ordering: `[0.8415, 0.5403, 0.0998, 0.9950, 0.0100, 0.99995, 0.0010, 1.0000]`.)
Verified in code: `SinusoidalEmbedding(8)(torch.tensor([1]))` → `[0.8415, 0.0998, 0.0100, 0.0010, 0.5403, 0.9950, 0.9999, 1.0000]`.

### Question 3 – Positional encoding vs. sinusoidal time embedding

**Same idea / same formula:** both map an integer (token position `pos` in a Transformer, noise level `t` in a diffusion model) to a vector of sines and cosines at geometrically spaced frequencies. This gives a smooth, bounded, unique code for every integer, in which nearby values have similar vectors, and which a network can easily read.

**Key difference in how they are used:**

- **Transformer:** the encoding varies *across the sequence* – each token gets a different vector according to its position – and it is **added to the token embeddings once at the input**, so that attention (which is otherwise permutation-invariant) knows the order of tokens.
- **Diffusion model:** the whole image shares **one** timestep `t`, so there is a single embedding per sample. It describes *how noisy the input is* (not where something is). It is usually passed through an MLP and **injected into every residual block of the UNet** (added to / used to scale the feature maps), conditioning the whole network on the noise level so one shared network can denoise at every step.

### Question 4 – Bottleneck resolution

Each stride-2 convolution halves the height and width:

$$64 \times 64 \;\xrightarrow{\ \div 2\ }\; 32 \times 32 \;\xrightarrow{\ \div 2\ }\; 16 \times 16 \;\xrightarrow{\ \div 2\ }\; 8 \times 8$$

In general $64 / 2^3 = 8$, so the bottleneck is **8 × 8** (number of channels depends on the architecture).

### Question 5 – What the UNet outputs and how the loss is computed

During training we take a clean image $x_0$, sample $t \sim \mathcal U\{1,\dots,T\}$ and noise $\varepsilon \sim \mathcal N(0, I)$, and form the noisy image in closed form:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \qquad \bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$$

The UNet receives $(x_t, t)$ and outputs $\varepsilon_\theta(x_t, t)$, a tensor with **the same shape as the image** that is its **prediction of the noise $\varepsilon$** that was added (not the clean image). The loss is the mean squared error between the true and predicted noise:

$$\mathcal L(\theta) = \mathbb E_{x_0,t,\varepsilon}\big[\,\lVert \varepsilon - \varepsilon_\theta(x_t,t)\rVert^2\,\big]$$

At sampling time, the predicted noise is used to step backwards: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t)\big) + \sigma_t z$ (or $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta)/\sqrt{\bar\alpha_t}$ for DDIM). See `train_diffusion` in `helper_lib/trainer.py` and `sample_diffusion` in `helper_lib/generator.py`.

---

## Part 3: Building Blocks of an Energy Model

### Question 6 – Basic gradient calculations

Output of the given code:

```
x.grad = tensor([7.])
```

**a)** $\dfrac{dy}{dx} = 2x + 3$. At $x = 2$: $2\cdot 2 + 3 = \mathbf{7}$, matching `x.grad = tensor([7.])`.

**b)** With `requires_grad=False`, no computational graph is recorded, so `y` has no `grad_fn` and `y.backward()` raises:

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

(and `x.grad` stays `None`).

**c)** No. The default is `requires_grad=False` (documentation of `torch.tensor`: *"requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False."*). `torch.tensor([2.0]).requires_grad` → `False`.

### Question 7 – Introduce weights

**a)** Output:

```
x.grad = tensor([7.])
w.grad = None
```

`w` was created without `requires_grad=True`, so autograd treats it as a constant: it is not a leaf of the graph that needs a gradient, and `backward()` never computes or stores one for it, so `w.grad` is `None`. (`x.grad` is again $2w_0x + w_1 = 2\cdot1\cdot2 + 3 = 7$.)

**b)** Mark `w` as requiring gradients:

```python
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True)

y = w[0] * x**2 + w[1] * x
y.backward()

print("x.grad =", x.grad)   # tensor([7.])
print("w.grad =", w.grad)   # tensor([4., 2.])
```

Check: $\partial y/\partial w_0 = x^2 = 4$, $\partial y/\partial w_1 = x = 2$. (If we only want the weights' gradient – as when training – we could also leave `x` with `requires_grad=False`.)

**c)** No – same as 6c: `torch.tensor` defaults to `requires_grad=False`, so gradients are only tracked for tensors that explicitly set `requires_grad=True` (or that are computed from such tensors). `nn.Parameter` objects inside `nn.Module`s set it to `True` automatically, which is why we never had to do this for network weights before.

### Question 8 – Breaking the graph

`y.detach()` returns a new tensor that shares data with `y` but is **cut out of the computational graph** (`requires_grad=False`, no `grad_fn`). Therefore `w = z * 2` doesn't depend on `x` as far as autograd knows, `w` has no `grad_fn`, and `w.backward()` raises
`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`.

Fix – keep `z` connected to the graph (don't detach; use `clone()` if a separate tensor is wanted):

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.clone()        # or simply z = y   – still part of the graph
w = z * 2

w.backward()
print(x.grad)        # tensor([6.])   dw/dx = 2 * 3 = 6
```

(Note: `z = y.detach().requires_grad_()` would *not* fix it – `backward()` would then run, but the gradient would stop at `z` and `x.grad` would stay `None`.)
This is exactly the trick used in the GAN discriminator step (`fake.detach()` so the generator is not updated) and, in reverse, in Langevin sampling (`x.detach().requires_grad_(True)` so the gradient goes to the image, not to earlier computations).

### Question 9 – Gradient accumulation

Output:

```
After first backward: x.grad = tensor([2.])
After second backward: x.grad = tensor([5.])
```

PyTorch **accumulates** (adds) gradients into `.grad` instead of overwriting them. The second call adds $d(3x)/dx = 3$ to the existing 2, giving 5 rather than 3.

To avoid it, reset the gradient before each new backward pass:

```python
x.grad.zero_()          # or x.grad = None
y2 = x * 3
y2.backward()
print(x.grad)           # tensor([3.])
```

In a training loop this is what `optimizer.zero_grad()` does before every `loss.backward()`. In Langevin sampling we use `torch.autograd.grad(energy.sum(), x)`, which *returns* the gradient instead of accumulating it into `.grad`, and we freeze the model weights (`requires_grad_(False)`) so no gradients pile up on the parameters (see `langevin_sample` in `helper_lib/generator.py`).
