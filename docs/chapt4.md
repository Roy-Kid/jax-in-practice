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

# 不同2: 越界行为

在numpy中

```python
a = np.arange(5)[6]
```

但是在jax.numpy中, 考虑到实际的计算会下放到加速硬件上进行, 不会拉起终止程序的错误. 当通过越界的索引更新元素时, 默认操作将跳过这个更新; 如果是索引越界的元素, 将会返回边界上的值. 

```python
a = jnp.arange(5)
a[6]
DeviceArray(4, dtype=int32)
```
归根结底, 越界索引是一种"不稳定"的操作, 应该把它看成未定义行为. JAX提供了一套索引语法`.at[]`来操作数组

```python
>>> x = jnp.arange(5.0)
>>> x
DeviceArray([0., 1., 2., 3., 4.], dtype=float32)
>>> x.at[2].add(10)
DeviceArray([ 0.,  1., 12.,  3.,  4.], dtype=float32)
>>> x.at[10].add(10)  # out-of-bounds indices are ignored
DeviceArray([0., 1., 2., 3., 4.], dtype=float32)
>>> x.at[20].add(10, mode='clip')
DeviceArray([ 0.,  1.,  2.,  3., 14.], dtype=float32)
>>> x.at[2].get()
DeviceArray(2., dtype=float32)
>>> x.at[20].get()  # out-of-bounds indices clipped
DeviceArray(4., dtype=float32)
>>> x.at[20].get(mode='fill')  # out-of-bounds indices filled with NaN
DeviceArray(nan, dtype=float32)
>>> x.at[20].get(mode='fill', fill_value=-1)  # custom fill value
DeviceArray(-1., dtype=float32)
```
`add`等操作提供了一个mode选项, 有三种可选. 默认行为是`promise_in_bounds`, 更新元素时的越界跳过, 索引越界元素返回边界值. `clip`