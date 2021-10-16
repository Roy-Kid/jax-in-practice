# JAX.numpy as jnp

JAX最吸引人的地方在于它的数据结构和numpy语法完全一致. 与numpy.ndarray对标的是jax.numpy.DeviceArray, 它是整个jax的核心数据对象. 虽然在语法上与两者完全一致, 但是背后的行为有很大的差别, 甚至影响到程序设计的思路和编程的模式.

DeviceArray的基类是jaxlib.xla_extension.DeviceArray, 有时候你会在报错中看到.

我们通常不直接初始化DeviceArray, 而是通过jax.numpy的array(), arange(), linspace()创建.

## 不同1: JAX array是不可变的

在numpy中, 我们想改变一个array的某个元素的值, 可以直接

```python

nparray[x] = y

```
但是这个操作在jnp中是不被允许的

```python
jnparray[x] = y

TypeError: '<class 'jaxlib.xla_extension.DeviceArray'>' object does not support item assignment. JAX arrays are immutable. Instead of ``x[idx] = y``, use ``x = x.at[idx].set(y)`` or another .at[] method: https://jax.readthedocs.io/en/latest/jax.ops.html
```

如报错所说, 我们需要用at[]这种辅助方法来代替直接的索引. 让我们来试一下, 并比较一下操作所需要的时间

```python

>>> nparray = np.empty((1000, 1000))
>>> %time nparray[123, 456] = 1

CPU times: user 7 µs, sys: 0 ns, total: 7 µs
Wall time: 9.3 µs

>>> jnparray = jnp.empty((1000, 1000))
>>> %time jnparray = jnparray.at[123, 456].set(1)

CPU times: user 0 ns, sys: 3.27 ms, total: 3.27 ms
Wall time: 2.96 ms
```

两者操作的时间相差了几千倍. 原因有两点, 第一, DeviceArray正如其名, 它要储存在硬件上, 例如发送到显存中, 因此把它取回主机需要时间; 第二, 由于不能进行原位操作, 因此需要将其复制一份返回到新变量中. 这意味着对数组的逐元素修改是巨大的时间开支. 

好在我们有传说中的jit(just-in-time), 在编译之后, 这种非原位的操作会优化为原位操作

```python
def jnp_set(x):
    return x.at[123, 456].set(1)

%time jit_jnp_set = jit(jnp_set)

CPU times: user 85 µs, sys: 0 ns, total: 85 µs
Wall time: 89.4 µs

%time jit_jnp_set(jnparray)
CPU times: user 19.5 ms, sys: 11.5 ms, total: 31 ms
Wall time: 37.9 ms

```

完全没有变快啊! 不过可以注意到, jit过程花的时间微乎其微, 但是第一次执行的时间却很长. 章节的最后, 我们将解释异步同调(Asynchronous dispatch).

# 不同2: Static 与 Traced

JAX是不鼓励对变量进行修改的, 因为原位修改以后, JAX就难以追踪对变量进行的运算. 回忆第二章所讲的__, 每一次操作, 或者说每一个算子所用到变量上, 总是要记录tangent或者adjoint. 如此就要引入两个概念, static和traced. 

static指的是能在jit编译时确定下来的变量(evaluated at compile-time). 

```python

def square(x):
    return x**2
jit_square = jit(square)
```
当我们对一个函数进行jit的时候, 其中的数值能够即时地确定下来, 而不是在调用这个函数的时候才能确定, 这个就成为编译时确定的. 因为它不随输入的改变而改变, 所以称为static. 我们可以使用`jax.make_jaxpr`看一看编译完的状态

```python
jax.make_jaxpr(square)(jnp.array([1., 2.,]))

{ lambda  ; a.
  let b = integer_pow[ y=2 ] a
  in (b,) }
```
可以看到编译的结果与输入`a`的值无关, 而直接把`2`确定了下来. 因此, 如果说函数中一个值与输入有关, 就会出现

```python
@jit
def f(x):
  return x.reshape(jnp.array(x.shape).prod())

x = jnp.ones((2, 3))
f(x)

---------------------------------------------------------------------------
ConcretizationTypeError                   Traceback (most recent call last)
<ipython-input-26-5fa933a68063> in <module>()
      7 
      8 x = jnp.ones((2, 3))
----> 9 f(x)

ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected.

The error arose in jax.numpy.reshape.
While tracing the function f at <ipython-input-26-5fa933a68063>:4, this value became a tracer due to JAX operations on these lines:

  operation c:int32[] = reduce_prod[ axes=(0,) ] b:int32[2]
    from line <ipython-input-26-5fa933a68063>:6 (f)

See https://jax.readthedocs.io/en/latest/faq.html#abstract-tracer-value-encountered-where-concrete-value-is-expected-error for more information.

Encountered tracer value: Traced<ShapedArray(int32[])>with<DynamicJaxprTrace(level=0/1)>

```
错误告诉你`jax.numpy.reshape`在编译过程中是一个tracer, 而不是static. 这就意味着它在编译的时候无法将这个操作确定下来. 我们看看其中都是什么变量

```python
@jit
def f(x):
  print(f"x = {x}")
  print(f"x.shape = {x.shape}")
  print(f"jnp.array(x.shape).prod() = {jnp.array(x.shape).prod()}")
  # comment this out to avoid the error:
  # return x.reshape(jnp.array(x.shape).prod())

f(x)

x = Traced<ShapedArray(float32[2,3])>with<DynamicJaxprTrace(level=0/1)>
x.shape = (2, 3)
jnp.array(x.shape).prod() = Traced<ShapedArray(int32[])>with<DynamicJaxprTrace(level=0/1)>
```

传入的x是traced, `x.shape`是static. 但是, 当`jnp.array`和`jnp.prod`作用到这个静态变量上, 就会转换成traced变量. 而`reshape()`操作必须要求一个static的变量输入, 从而发生了错误. 解决这种冲突的办法就在`jnp`和`np`上. 既然JAX完全支持Numpy的API, 我们为什么还要把jnp和np区分开? 我们可以用numpy处理static, 在jit时进行优化, 用jax.numpy处理traced, 在运行时优化. 对于上面的函数, 我们可以写成

```python
from jax import jit
import jax.numpy as jnp
import numpy as np

@jit
def f(x):
  return x.reshape((np.prod(x.shape),))

f(x)

```

