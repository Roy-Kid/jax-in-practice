1. 求导多种形式
1. 导数, gradient, jacobian, hassian
1. gradient-vector products, hessian-vector prod
1. dual number, chain rule
1. finit differences
1. directional derivatives
1. adjoint

# 

假如一个小球和墙面通过弹簧相连, 只能在一条线上运动, 其能量满足胡克定律:

$$
    U(x) = \frac{1}{2}kx^2
$$

运用小学二年级的导数的知识, 我们可以写出任意位置的力的表达式:

$$
    F(x) = -\frac{dU}{dx} = kx 
$$

如果这个小球可以在空间中任意位置运动, 我们只需要把坐标$x$写作位置向量$\mathbf{x}$:
$$
    U(\mathbf{x}) = \frac{1}{2}k\mathbf{x}^2
$$
多个小球又可以写成矩阵$\mathbf{X}$的形式,.
那么如何求导
$$
    F(\mathbf{x}) = -\frac{dU}{d\mathbf{x}}
$$
和
$$
    F(\mathbf{X}) = -\frac{dU}{d\mathbf{X}}
$$

## 分类

给定一个输入, 对输入中的元素应用一个对应法则$f$, 得到另一个输出, 我们称为函数. 函数的输入和输出都可以是标量, 向量和矩阵, 两两组合一共有九种形式:

|    |    |    |
|--- |--- | ---|
|$f(x)\to x$| $f(\mathbf{x})\to x$|$f(\mathbf{X})\to x$|
|$f(x)\to \mathbf x$| $f(\mathbf{x})\to \mathbf x$|$f(\mathbf{X})\to \mathbf x$|
|$f(x)\to \mathbf X$| $f(\mathbf{x})\to \mathbf X$|$f(\mathbf{X})\to \mathbf X$|

虽然有九种形式, 但对于求导来说基本形式只有第一行三种. 其它行的可以看作是函数作用了多次. 例如:

$$
    f(\mathbf x) = \left[ \frac{\partial f_1}{\partial x_1}, \frac{\partial f_2}{\partial x_2} \right]
$$

### 函数的输出是标量

1. 输入是标量:

    这就是最普通的一元函数

2. 输入是向量:

    $$
    \begin{aligned}
        &\mathbf{x} = [x_1, x_2, x_3] \\

        &f(\mathbf{x}) = x_1^2+ x_2^2+ x_3^2\\

        &\frac{df(\mathbf{x})}{d\mathbf{x}} =  [\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \frac{\partial f}{\partial x_3} ] = [2x_1, 2x_2, 2x_3]\\
    \end{aligned}
    $$

3. 输入的是矩阵:

    我们可以看成多个行/列向量同时传入一个函数
    
    $$
    \begin{aligned}
        &\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3] \\

        &f(\mathbf{X}) = [f(\mathbf{x_1}), f(\mathbf x_2), f(\mathbf x_3)] = \Sigma x_{ij}^2\\

        &\frac{df(\mathbf{X})}{d\mathbf{X}} = 
    \end{aligned}
    $$
$$
\left[\begin{array}{lll}
\frac{\partial f_{1}}{\partial x_{1}} & \frac{\partial f_{1}}{\partial x_{2}} & \frac{\partial f_{1}}{\partial x_{3}} \\
\frac{\partial f_{2}}{\partial x_{1}} & \frac{\partial f_{2}}{\partial x_{2}} & \frac{\partial f_{2}}{\partial x_{3}}
\end{array}\right] \\
$$

### 函数的输出是向量

相当于在每个元素上都有一个对应法则$f_i$, 然后可以应用同样的求导方法

$$
    f(x) = \mathbf{x} = [\mathbf x_1, \mathbf x_2, \mathbf x_3] = [f_1(x), f_2(x), f_3(x)]
$$


## 梯度向量积

有一个光滑的函数$f:\mathbb R^N \to \mathbb R$, 在$x\in \mathbb R^N$沿着任意方向向量$v$的导数定义为:

$$
\nabla_{v} f(x)=\lim _{\epsilon \rightarrow 0} \frac{f(x+\epsilon \cdot v)-f(x)}{\epsilon}
$$

这个定义等价于这一点的梯度乘以这个方向向量:

$$
\nabla_{v} f(x)=\nabla f(x) \cdot v=\sum_{n=1}^{N} f_{n}(x) \cdot v_{n}=\sum_{n=1}^{N} \frac{\partial f(x)}{\partial x_{n}} \cdot v_{n}
$$

这个向量乘法是保形的, 因为根据定义$\nabla f(x)$是行向量, 而$v$是列向量. $f$在$x$沿着单位向量$u$的导数称为方向导数, 因为这展现的是$f$在$u$方向上的变化趋势. 
