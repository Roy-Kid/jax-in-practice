1. forward mode
1. reserve mode
1. nested forward mode
1. 
1. 

## 前向模式

前向模式(forward mode)又称为正切模式(tangent mode), 是一种高效的求导单数入多输出的光滑函数方法. 

假设$x\in \mathbb R$ 和$v=f(u)$, 我们在表达式上记一个点以和一般变量分开:

$$
    \dot u = \frac{\partial}{\partial x}u
$$

$\dot u$称为$u$关于$x$的正切; $x$在符号中是隐式的, 但在多个加点达式的表达式中被假定为相同的$x$.
例如, 如果$-v=u$, 根据链式法则,

$$
    \dot v = \frac{\partial}{\partial x}v = -\frac{\partial}{\partial x}u = -\dot u
$$

类似地, 如果我们有$v = exp(u)$, 那么

$$
    \dot v = \frac{\partial}{\partial x}v = -\frac{\partial}{\partial x}\text{exp}(u) = \text{exp}(u) \cdot \dot u
$$

正向模式下的导数传播与多元函数的工作方式相同. 例如, 如果$y=u\cdot v$, 那么将应用一般的乘积求导法则: 

$$
\dot{y}=\frac{\partial}{\partial x} u \cdot v=\left(\frac{\partial}{\partial x} u\right) \cdot v+u \cdot\left(\frac{\partial}{\partial x} v\right)=\dot{u} \cdot v+u \cdot \dot{v}
$$

## 二元数
为了表示方便, 这里引入二元数的概念(dual number). 一个表达式及其导数我们记作

$$
\langle u, \dot{u}\rangle
$$
相反数:
$$
-\langle u, \dot{u}\rangle=\langle-u,-\dot{u}\rangle
$$
求和:
$$
\langle u, \dot{u}\rangle + \langle v, \dot{v}\rangle=\langle u+v, \dot{u} + \dot v \rangle
$$
作差:
$$
\langle u, \dot{u}\rangle - \langle v, \dot{v}\rangle=\langle u-v, \dot{u} - \dot v \rangle
$$
乘积:
$$
\langle u, \dot{u}\rangle \cdot \langle v, \dot{v}\rangle=\langle u\cdot v, \dot{u} \cdot \dot v \rangle
$$
相除:
$$
\langle u, \dot{u}\rangle / \langle v, \dot{v}\rangle=\langle u / v, \dot{u} / v - u/v^2 \cdot \dot v \rangle
$$
取对数:
$$
log\langle u, \dot{u}\rangle =\langle \text{log} u, \frac{\dot{u}}{u}  \rangle
$$

## 梯度向量积的二元数表示

## 方向导数的二元数表示

## 反向模式

反向模式同样是自动微分算法, 也叫做共轭方法, 在多输入单输出时具有很高的计算效率. 
