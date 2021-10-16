# JAX

JAX是谷歌开发的高性能数值计算和自动微分库, 底层使用了加速线性代数编译器(XLA, Accelerated Linear Algebra)使代码可以运行在CPU, GPU和TPU上, 同时实现微分, 矢量化和并行化, 即时编译到GPU/TPU等功能. 在科学计算方向, Numpy有着独特的优势, 底层, 灵活, API稳定且为大家熟悉(与MATLAB一脉相承), 深受学术界青睐. JAX将Numpy的以上优势与硬件加速结合, 将Python的科学计算提高到一个新的高度.

JAX提供了三大功能, 微分, jit和矢量与并行. JAX可以对原生的Python/Numpy函数求导, 兼容循环, 分支控制, 递归和闭包. 并且可以求得高阶导数。它支持通过梯度的反向模式微分（又称反向传播）和正向模式微分，两者可以任意顺序组合。

```python
from jax import grad
import jax.numpy as jnp

def tanh(x):  # Define a function
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)  # Obtain its gradient function
print(grad_tanh(1.0))   # Evaluate it at x = 1.0
# prints 0.4199743
print(grad(grad(grad(tanh)))(1.0))
# prints 0.62162673
```

JAX使用XLA对Python代码进行编译, 使其可以运行在GPU和TPU上, 这个称为Just-in-time编译在执行操作的时候默认执行, 也可以手动编译函数使其得到XLA的优化. 编译和求导的过程可以任意组合, 用Python也可以实现复杂的算法且获得高性能. 

```python
import jax.numpy as jnp
from jax import jit

def slow_f(x):
  # Element-wise ops see a large benefit from fusion
  return x * x + x * 2.0

x = jnp.ones((5000, 5000))
fast_f = jit(slow_f)
%timeit -n10 -r3 fast_f(x)  # ~ 4.5 ms / loop on Titan X
%timeit -n10 -r3 slow_f(x)  # ~ 14.5 ms / loop (also on GPU via JAX)
```

甚至可以通过pmap等方法, 实现多GPU或TPU并行计算. 

`vmap`是矢量化映射方法, 调用时只需要指定需要映射的数组轴, 就可以实现批处理. 它不是在函数的外部进行循环, 而是将循环下沉到函数的内部操作中实现高性能的计算. 使用vmap避免了在编写函数的时候需要考虑数组维度等问题. 例如要计算两个元素之间绝对值

```python

def absstruct(x, y):
    return abs(y-x)

# a.shape: (1, )
# b.shape: (1, )

ans = absstruct(b - a)

# a.shape: (1, )
# b.shape: (N, 1)

ans = vmap(absstruct, (0, None))(b, a)

```

## 随机数

在v1.17之后的numpy中, 生成一个5000*5000的均匀随机矩阵并与自己相乘

```python
# 建议使用如下操作
from numpy.random import default_rng
rng = default_rng()
vals = rng.standard_normal((5000, 5000))

%time vals@vals
CPU times: user 9.44 s, sys: 386 ms, total: 9.82 s
Wall time: 1.23 s

```

在JAX中
```python
import jax.numpy as jnp  # JAX版的numpy
from jax import random  

x = random.uniform(random.PRNGKey(0), (5000, 5000))
%timeit jnp.matmul(x, x)
# without GPU
CPU times: user 4.71 s, sys: 0 ns, total: 4.71 s
Wall time: 591 ms 

# with GPU
```

## Jax.numpy

Jax.numpy提供了一套与numpy完全相同的API, 对底层的jax.lax进行了封装. 

> 然而, 即便API相同, 两者的表现和行为完全不一样, 我们会在下一章中详细阐述其中的差异. 

所有的lax操作都是通过底层的XLA(the Accelerated Linear Algebra compiler)实现的. 查看JAX的源代码, 会发现所有的操作都会调用jax.lax. jax.lax可以看成是更加严格但是更强大的多维数组操作接口. 

例如, 我们可以不考虑类型, 直接进行两个数的加法
```python
jnp.add(1, 1.0)
```
jax.numpy会隐式地处理这两种不同的类型. 但是如果直接调用lax, 则需要手动处理类型的转换
```python
lax.add(jnp.float32(1), 1.0) # 正确
lax.add(1, 1.0)

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-10-63245925fccf> in <module>()
      1 from jax import lax
----> 2 lax.add(1, 1.0)  # jax.lax API requires explicit type promotion.

TypeError: add requires arguments to have the same dtypes, got int32, float32.
```

# 异步调度

JAX使用了异步调度(asynchronous dispatch)以提升Python的性能. 

> 异步：不等任务执行完，直接执行下一个任务

举一个矩阵乘法的例子: 

```python

key = jax.random.PRNGKey(41)
jnparr = jax.random.uniform(key, (1000, 1000))
jnp.dot(jnparr, jnparr) + 3

```
当执行到`jnp.dot()`的时候，Jax并不会等待这个运算计算结束，而是返回一个`DeviceArray`类型的对象，然后继续执行Python代码。这个对象将在加速处理器上进行计算，但我们可以不等其计算完成就可以获取到它的类型和形状，甚至像上面一样可以传给下一个JAX计算式。只有我们在主机端实际检查数组的值时，例如`print()`或者`np.asarray()`转换为Numpy数组时才会阻塞等待计算完成。

```python
>>> def f(x):
>>>     # do something
>>>     return jnp.sum(x)

>>> jitted_f = jit(f)
    # do other things
>>> ans = jitted_f(x)
CPU times: user 6.89 ms, sys: 8.63 ms, total: 15.5 ms
Wall time: 14.2 ms
>>> ans = jitted_f(x)
CPU times: user 338 µs, sys: 103 µs, total: 441 µs
Wall time: 263 µs
```

同理，`jit()`的过程也是异步调度的。当jit一个函数的时候会返回一个`CompiledFunction`对象，但是并没有被编译，而是直到这个函数第一次被执行的时候才会被编译。

这种惰性计算或者异步调度的方式可以让Python代码领先于加速器进度，使得Python不再是瓶颈。换言之，将逻辑命令与实际的计算相分离，使Python不需要在主机上检查对应的输出，Python就可以不断地将任务加入队列，避免加速设备等待。

做一个测试：

```python
>>> %time jnp.dot(x, x)

# --- output ---
CPU times: user 267 µs, sys: 93 µs, total: 360 µs
Wall time: 269 µs
```
269μs对于1000x1000的矩阵乘法有些过于断了！ 然而这只是执行异步调度所需要的时间。如果测试真正乘法所需要的时间，我们需要访问它的值。例如转换为Numpy：
```python
>>> %time np.asarray(jnp.dot(x, x))
# --- output ---
CPU times: user 61.1 ms, sys: 0 ns, total: 61.1 ms
Wall time: 8.09 ms
```
或者是使用`block_until_ready()`阻塞：
```python
>>> %time jnp.dot(x, x).block_until_ready()  
# --- output ---
CPU times: user 50.3 ms, sys: 928 µs, total: 51.2 ms
Wall time: 4.92 ms
```
阻塞但不将结果传回主机端通常会更快一些，也是做基准测试的最佳选择。

## grad



## to JIT or not to JIT, it's a question

JAX的时候绕不开JIT. 即便你不考虑运行时间只要求一个正确结果, 那很多JAX方法都会隐式地进行JIT. 可以说, JAX的编程离不开三件事, jnp, grad 和jit. 

JIT在程序启动后运行, 并将字节码(bytecode)实时(on the fly / just-in-time)转换成更快的形式, 通常是主机的CPU原生指令集形式. JIT可以访问动态的运行时的信息, 并且进行更好地优化, 标准编译器则不能, 也就是说标准编译器不能获取在编译结束之后传入的数据的信息(如数组大小等). 与运行之**前**就要进行编译的传统编译器不同, JIT会在程序启动**后**再进行编译, 或者是按需编译需要加速的代码片段. 

例如以下代码

```python
import jax.numpy as jnp

def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)
```

JAX中使用`jax.jit()`将一个函数转换为jit版本

```Python
from jax import jit
norm_compiled = jit(norm)
```

jit版本的结果和原始版本一样

```python
np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
np.allclose(norm(X), norm_compiled(X), atol=1E-6)

True
```

通过包括操作合并, 避免分配临时数组和其他tricks在内的操作, jit之后的速度将会大大加快. 由于异步同调的问题, 这里需要手动阻断等待结果计算完成才能获得真实的计算时间

```python
%timeit norm(X).block_until_ready()
%timeit norm_compiled(X).block_until_ready()

100 loops, best of 3: 4.3 ms per loop
1000 loops, best of 3: 452 µs per loop
```
