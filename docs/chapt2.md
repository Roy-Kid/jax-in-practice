# 第二章 自动微分



## 手动微分

手动微分本质上讲是一种基于规则(rule-based)的方法. 计算时会定义大量的运算符以实现自动微分的功能. 

```python
class Variable:

    def __init__(self, value):
        self.value = value
        self.diff = 0

    def __add__(self, value):
        self.diff = self.diff
        return self.value + value

    def __multi__(self, value):
        self.diff = value
        return self.value * value
```

虽然直观且易于理解, 但是对于复杂的数学过程需要编写大量的规则且效率低下. 

## 符号推导法

python中的sympy实现了一套符号推导的机制, 可以从符号的角度推导导函数, 可以说是"自动的"手动推导. 
```python
import numpy as np
import sympy
from sympy import sin
from sympy.abc import x, y

def f(x, y):
    return sin(x)*y + y**3

z = f(x, y)
print('z = ', z)
out: z = y**3 + y*sin(x)

dzdx = z.diff(x)
print('dzdx = ', dzdx)
out: y*cos(x)

```
符号推导完全是基于表达式的, 因此几乎可以得到和导导函数一样精度. 不过这种方法也不是十全十美的, 其中一个问题叫做"表达式膨胀(expression swell)". 也就是说, 推导出来的导函数长度可能相对于原函数有指数级增加. 这是由求导规则带来的

$$
\begin{align}
    &h(x) = f(x)g(x) \\
    &h^\prime(x) = f^\prime(x)g(x) + f(x)g^\prime(x) \\
\end{align}
$$
例如如上的乘法导数规则会带来很多重复的计算. 其次, 符号推导要求函数形式是封闭形式的, 不能使用循环, 判断或递归等形式.

## 数值微分法

数值求导法(Numerical Differentiation)又可以叫做有限差分法(Finite Differentiation). 这一方法直接来自于微分的定义


$$
    \frac{\partial f}{\partial x_i}=\lim _{\varepsilon \rightarrow 0} \frac{f(x+\varepsilon \cdot e_i)-f(x)}{\varepsilon}
$$

$e_i$为第$i$个元素为1的单位向量按照定义, 某一点的导数是在这一点的微小该变量. 当这一改变量趋近于0时, 就得到了导数. 例如, 计算$x=3$在函数$f(x)=x^2$处的导数:

```python
>>> def square(x: jnp.ndarray)->jnp.ndarray:
        return jnp.sum(x**2)
>>> assert 6.0 == grad(square)(x=3.0)
out: DeviceArray(True, dtype=bool)
>>> eps = 1e-4
>>> assert jnp.isclose((square(3.0+eps) - square(3.0))/eps, 6.0001)
out: DeviceArray(True, dtype=bool)
```
这个过程很自然地带来两个问题, 即微小量取值和相应的误差. 误差又来源于两个方面, 一方面是数学上引入的误差, 另一方面是计算机原理上的误差. 在第三章章末, 我们会专门讲解计算机精度的问题. 

> 引入https://zhuanlan.zhihu.com/p/109755675的推导, 包括前向/后向和中心差分

## 自动微分算法
自动微分算法可以达到和符号推导一样的精度, 而且更加灵活. 其核心原理是追踪每一个基本运算, 例如加减乘除, 然后运用链式规则确定最终结果. 自动微分算法可以分为两类, 一种是前向模式(forward mode), 另一种是反向模式(reverse mode). 

### 前向模式

一般计算一个函数, 都是输入一个原始值, 流经整个计算过程得到输出. 前向模式使用了二元数以同时追踪求值计算和导数计算的中间变量.  $x$的二元数写作$\left< x, \dot x\right>$, 即$x$及其导数$\dot x$. 二元数之间可以进行运算, 遵从值与值计算, 导数与导数计算. 例如

乘积:
$$
\langle u, \dot{u}\rangle \cdot \langle v, \dot{v}\rangle=\langle u\cdot v, \dot{u} \cdot \dot v \rangle
$$

例如有一个函数
$$
    f(x_1, x_2) = \left[ \sin{\frac{x_1}{x_2}} + \frac{x_1}{x_2} - e^{x_2}\right] \times \left[ \frac{x_1}{x_2} - e^{x_2} \right]
$$

用二元数追踪其计算过程有

$$

    x = [3.7 12.6]^\mathsf T
    v = []

$$

# TODO: 梯度向量积