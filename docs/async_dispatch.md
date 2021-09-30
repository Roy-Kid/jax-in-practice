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