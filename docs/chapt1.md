# 第一章: 导数

## 函数

从数学的角度上讲, 函数是一种对应关系, 输入值集合中的每项元素皆能对应​​唯一一项输出值集合中的元素. 例如实数$x$对应到其平方$x^{2}$的关系就是一个函数，若以$3$作为此函数的输入值，所得的输出值便是$9$, 这个过程可以用数学语言表达为$f:\mathbb R \to \mathbb R$。 从编程的角度来看, 函数代表一种操作, 对一个输入的值, 经过一些操作之后, 输出另一个值

```python
>>> def square(x: float)->float:
        return x**2
>>> assert 9.0 == square(x=3.0)
out: True
```
如果这个函数在定义域上连续可导, 称其为光滑函数. 如果没有特别强调, 本书中的函数都是光滑函数的，且可以多次求导. 

## 导数
对于一个变量的光滑函数$f:\mathbb R \to \mathbb R$, 它的导函数可以写作$f^\prime: \mathbb R\to \mathbb R$. 这个记号意味着它将一个属于实数集的变量$x\in \mathbb R$通过对应法则$f$或$f^\prime$映射到另一个实数集上. 用定义可以写作
$$
    f^\prime(x)=\lim _{\varepsilon \rightarrow 0} \frac{f(x+\varepsilon \cdot v)-f(x)}{\varepsilon}
$$

导数衡量了在变量微小变化的情况下, 函数值的变化情况. 整理表达式可得

$$
    \lim_{\varepsilon \to 0} \frac{f(x+\varepsilon)-(f(x)+\varepsilon \cdot f^\prime(x))}{\varepsilon} = 0
$$

因此, 当$\varepsilon \to 0$的时候, 得到微分的定义

$$
    f(x+\varepsilon) \approx f(x) + \varepsilon \cdot f^\prime(x)
$$

通俗来说, 微分是一个函数在其自变量做无穷小变化时函数值的变化. 导数是曲线在那个点的切线斜率，而微分是那个切线的一元线性方程。 微分的几何意义是用局部切线段近似代替曲线段，即非线性函数局部线性化。例如, 我们对求平方的函数进行求导

```python
>>> def square(x: jnp.ndarray)->jnp.ndarray:
        return jnp.sum(x**2)
>>> assert 6.0 == grad(square)(x=3.0)
out: DeviceArray(True, dtype=bool)
>>> eps = 1e-4
>>> assert jnp.isclose((square(3.0+eps) - square(3.0))/eps, 6.0001)
out: DeviceArray(True, dtype=bool)
```

### 多元函数



### 偏导数

有$N$个变量的光滑函数$f:\mathbb R^N \to \mathbb R$, 对其中一个变量求导

$$
    \frac{\partial}{\partial x_n}f(x)=g^\prime(x_n)
$$

函数$g:\mathbb R \to \mathbb R$是仅对其中一个变量求导的函数, 写作

$$
    g(u) = f(x_1, x_2, ..., u, x_{n+1}, x_N)
$$

即, 对函数$f(x_n)$我们仅仅对其中一个变量求导, 而保持其他值$x_1, x_2, ..., x_{n+1}, x_N$为常数不变.

## 梯度

如果一个光滑函数的输入是$N$维向量, 记作$f: \mathbb R^N \to R$, 它的梯度为$\nabla f:\mathbb R^N \to \mathbb R^N$
$$
    \nabla f(x)=\left [ \frac{\partial}{\partial x_1}  \cdots \frac{\partial}{\partial x_N} \right ]
$$
梯度的转置记作
$$
\nabla^{\top} f(x)=(\nabla f(x))^{\top}=\left[\begin{array}{c}
\frac{\partial}{\partial x_{1}} f(x) \\
\vdots \\
\frac{\partial}{\partial x_{N}} f(x)
\end{array}\right]
$$

## 雅可比矩阵

考虑一个光滑的的多变量函数$f:\mathbb R^N \to \mathbb R^M$, 它的雅可比函数写作$J_f:\mathbb R^N\to (\mathbb R^N \times \mathbb R^M)$, 输入一个$N$维的向量, 输出一个$M \times N$的矩阵, 其中每个元素都是某一个分量的导数

$$
\mathrm{J}_{f}(x)=\frac{\partial}{\partial x_1}f(x)=\left[\begin{array}{c}
\nabla f_{1}(x) \\
\vdots \\
\nabla f_{M}(x)
\end{array}\right]=\left[\begin{array}{ccc}
\frac{\partial}{\partial x_{1}} f_{1}(x) & \cdots & \frac{\partial}{\partial x_{N}} f_{1}(x) \\
\vdots & \vdots & \vdots \\
\frac{\partial}{\partial x_{1}} f_{M}(x) & \cdots & \frac{\partial}{\partial x_{N}} f_{M}(x)
\end{array}\right]
$$

每一行都是对于输入向量的一个元素的梯度, 记作

$$
    f_m(x) = f(x[m])
$$

$[m]$意为从输出长度为$M$中选取第$m$个元素. 类似地, 整个雅可比矩阵也可以写成

$$
    J_f(x)[m,n] = \frac{\partial}{\partial x_n}f_m(x)
$$

## 海森矩阵

考虑一个输入为$N$输出为标量的多元函数$f:\mathbb R^N \to \mathbb R$. 海森函数$H_f$将输入映射到它的二阶导数矩阵, 可以用两次梯度记号表示

$$
\mathrm{H}_{f}(x)=\nabla \nabla^{\top} f(x)=\nabla\left[\begin{array}{c}
\frac{\partial}{\partial x_{1}} f(x) \\
\vdots \\
\frac{\partial}{\partial x_{N}} f(x)
\end{array}\right]=\left[\begin{array}{c}
\nabla \frac{\partial}{\partial x_{1}} f(x) \\
\vdots \\
\nabla \frac{\partial}{\partial x_{N}} f(x)
\end{array}\right]=\left[\begin{array}{ccc}
\frac{\partial^{2}}{\partial x_{1} \partial x_{1}} f(x) & \cdots & \frac{\partial^{2}}{\partial x_{1} \partial x_{N}} f(x) \\
\vdots & \vdots & \vdots \\
\frac{\partial^{2}}{\partial x_{N} \partial x_{1}} f(x) & \cdots & \frac{\partial^{2}}{\partial x_{N} \partial x_{N}} f(x)
\end{array}\right]
$$
如果逐个元素地表示, 海森对$x$是这样操作的
$$
\mathrm{H}_{f}(x)[m, n]=\frac{\partial^{2}}{\partial x_{m} \partial x_{n}} f(x)=\frac{\partial}{\partial x_{m}} \frac{\partial}{\partial x_{n}} f(x)
$$
海森矩阵是对称的, 有
$$
    H_f(x)[m,n] = H_f(x)[n,m]
$$
对角线是二阶偏导
$$
\mathrm{H}_{f}(x)[n, n]=\frac{\partial}{\partial x_{n}} \frac{\partial}{\partial x_{n}} f(x)=\frac{\partial^{2}}{\partial x_{n}^{2}} f(x)
$$