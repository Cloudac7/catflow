# 团簇结构分析

## 载入团簇结构

首先将需要分析的团簇轨迹文件读入到程序中，用户可以指定 `path` 参数为文件路径，
其他的参数请以可选参数形式传入。

```python
from catflow.analyzer.structure.cluster import Cluster

trajfile = "./dump.lammpstrj"
c = Cluster(trajfile, topology_format="LAMMPSDUMP", dt=0.0005)
```

!!! tip "提示"
    注意默认的格式为"XYZ"，如需要导入其他格式，请参考
    [MDAnalysis文档](https://userguide.mdanalysis.org/stable/formats/)。

或将 MDAnalysis 的一个 `Universe` 实例导入：

```
from MDAnalysis import Universe
u = Universe(trajfile, topology_format="LAMMPSDUMP", dt=0.0005))
d = Cluster.convert_universe(u)
```

## Lindemann Index 计算

为分析体系的相变性质，Lindemann等人[[1]](#1)提出了对结构随时间均方键长的演变进行分析来确定相变性质的方法，
一般计算公式如下：

$$< q_i >_{atoms} = \frac{1}{N(N-1)} \frac{\sqrt{<r^2_{ij}> - <r_{ij}>^2}}{<r_{ij}>}$$

这里参照 Welford 算法[[2]](#2)计算方差，简单实现了 Lindemann Index的计算方法，调用如下：

```python
lpf = c.lindemann_per_frames(u, select_cluster="name Pt")
```

即可得到关于整条轨迹Lindemann index的变化趋势，对 `lpf` 作图，可以帮助判断相变情况以及确定MD是否收敛。

注意这里 `lindemann_per_frames` 读入的是MDAnalysis中的Universe对象，通常用 `select_lang` 来指定需要对哪些原子进行分析，
即给定对应的Atom selection language来选取，
请参考[官方文档](https://userguide.mdanalysis.org/stable/selections.html)的说明。

### 拟合

为对团簇相变行为的温度依赖进行分析，常对得到的曲线进行拟合，可以采用如下函数，以dataframe形式输出拟合的曲线：

```python
import numpy as np

temperature = np.array([300., 400., 500., 600., 700.])
lindemann = np.array([0.05, 0.10, 0.20, 0.25, 0.32])
bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 400, 15.], 
          [np.inf, np.inf, np.inf, np.inf, 700., 100.])
df = fitting_lindemann_curve(temperature, lindemann, bounds, function='func2')
```

其中，

- `func1`函数形式为：
  $$ f(x) = b + (a - b) x + \frac{d}{1 + \exp({\frac{x - x_0}{\mathrm{d}x})}} + cx $$
- `func2`函数形式为
  $$ f(x) = \frac{ax+b}{1 + \exp({\frac{x - x_0}{\mathrm{d}x})}} + \frac{cx+d}{1 + \exp({\frac{x_0 - x}{\mathrm{d}x})}}$$

默认采用 `func2` 进行拟合，`bounds`中上下界的参数对应即为`a, b, c, d, x0, dx`的取值范围。

## References

<a id="1">[1]</a> F. A. Lindemann, Zeitschrift für, *Phys.* **1910**, 11, 609–612.

<a id="2">[2]</a> Donald E. Knuth, *The art of computer programming, volume 2 (3rd ed.)*: seminumerical algorithms, Addison-Wesley Longman Publishing Co, **1997**, 232.