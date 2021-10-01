---

最后会根据前面的几章的内容简单加一段引言

---

# 案例3："FAST"主动反射面的形状调节
> 部分内容改编自2021年高教社杯全国大学生数学建模竞赛A题

## 背景介绍
        中国天眼——500 米口径射电望远镜（_Five-hundred-meter Aperture Spherical radio Telescope_, 简称FAST）是我国具有自主知识产权的、目前世界上单口径最大、灵敏度最高的射电望远镜。位于中国贵州省黔南布依族苗族自治州境内，是中国国家“十一五”重大科技基础设施建设项目。从2011年3月25日动工兴建，到2020年1月11日通过中国国家验收工作正式开放运行，FAST的落成启用，对我国在科学前沿实现重大原创突破、加速创新驱动发展具有重要意义。
![chapter10-fast.png](https://cdn.nlark.com/yuque/0/2021/png/22279603/1632831959469-8570a062-dd28-4cac-acdb-680145a08aec.png#clientId=ubde56e6b-f8d3-4&from=ui&id=u0e778863&margin=%5Bobject%20Object%5D&name=chapter10-fast.png&originHeight=458&originWidth=821&originalType=binary&ratio=1&size=48312&status=done&style=none&taskId=ue2097169-7c4f-43b4-aabc-74a45a3976a)**图1**：钱宏亮. FAST 主动反射面支承结构理论与试验研究 [D]. 哈尔滨: 哈尔滨工业大学, 2007.
         
        FAST由`主动反射面`、信号接收系统（`馈源舱`）以及相关的控制、测量和`支承系统`组成，如**图1**所示，主动反射面是由`主索网`、`反射面板`、`下拉索`、`促动器`及支承结构等主要部件构成的一个可调节球面。主索网由柔性主索按照短程三角方格网格方式构成，用于支承反射面板，每个三角网格上安装一个反射面板，整个索网固定在周边支承结构上。每个主索节点都对应一个下拉索，下拉索的下端与固定在地表的促动器连接，实现对主索网形态的控制。反射面板之间留有一定的缝隙，能够确保反射面板在变位时不会被挤压、拉扯而变形。
        主动反射面分为`基准态`和`工作态`两个状态。基准态时反射面为半径R=300米、口径D=500米的球面（`基准球面`）；工作时反射面则需要被调节为一个300米口径的近似旋转抛物面（`工作抛物面`）。右下图之中，C点是基准球面的球心，馈源舱接收平面的中心只能够在基准球面同心的球面（焦面）上移动，不同的文献给出的两同心球面半径差(焦径比)的数值约在 F=0.466~0.467R 之间，馈源舱接受信号的有效区域为直径为1m的中心圆盘。我们需要通过调节部分反射面板的形状，以形成以直线SC为对称轴，以P为焦点的近似旋转抛物面，从而将来自目标天体的平行电磁波反射汇聚到馈源舱的有效区域。
![[chapter10] fast2.png](https://cdn.nlark.com/yuque/0/2021/png/22279603/1632843592030-00b18f4e-42e7-4165-96d3-b4bcd6f7a77b.png#clientId=ubde56e6b-f8d3-4&from=ui&height=251&id=uc2a2c64c&margin=%5Bobject%20Object%5D&name=%5Bchapter10%5D%20fast2.png&originHeight=826&originWidth=1305&originalType=binary&ratio=1&size=131081&status=done&style=none&taskId=u66e51381-f429-4e5e-afbd-c0723abecb7&width=397.3333740234375)![FAST3.png](https://cdn.nlark.com/yuque/0/2021/png/22279603/1632844009127-a75f9157-c643-46f0-89f6-309febb00beb.png#clientId=ubde56e6b-f8d3-4&from=ui&height=291&id=u0e1d202c&margin=%5Bobject%20Object%5D&name=FAST3.png&originHeight=641&originWidth=750&originalType=binary&ratio=1&size=118177&status=done&style=none&taskId=u7cbbaab7-def0-4f0b-a982-7b475cfc25d&width=340.3333435058594)
> - 左图来源：巨型射电望远镜索网结构的优化分析与设计_姜鹏
> - 右图来源：2021年CUMCM竞赛试题A图4

​

        将反射面调节为工作抛物面是技术的关键所在，实际的工程之中，我们通过下拉索与促动器配合来完成工作抛物面的调节。促动器沿着基准球面径向安装，底端固定，但顶端可以进行径向的伸缩来完成下拉索的调节，从而控制反射面板的位置，最终形成理想抛物面。实际的工程还要求：

- 主索节点调节后相邻节点之间的距离变化幅度不要超过0.07%；
- 促动器径向伸缩的范围为-0.6~+0.6米；

        对于此类问题，通常在工程上我们会采用ANSYS等大型有元软件进行建模及数值模拟[ref-1]，在可接受的误差范围内给出各类我们所需要的参数。然而，在这个例子当中，通过引入可微分编程框架，您会发现我们将以相当简单的计算方式，在相当的精度之内，同大型软件一样给出每个促动器径向伸缩的距离、每个主索节点变位过程之中侧向的偏移量，甚至每根柔性主索当中受到应力的大小。**在深度学习之中，我们需要最小化的目标是通过某种方式定义的loss函数；而实际的物理体系，天然就会趋向于能量最小的状态，因此，我们不妨通过某种可微的方式定义出体系的能量，然后通过模拟能量最小化的过程，求解实际体系的状态**。
> ref-1:  巨型射电望远镜索网结构的优化分析与设计_姜鹏

        该章节之中我们提供的数据有：

- vertex.csv
   - 用于储存2226个主索节点的坐标的基准态(初态);  第一列数据为主索节点的标号**(Index)**，后三列分别储存主索节点初始状态下**（X，Y，Z）**的坐标；
- vertex_ground.csv
   - 用于储存2226个促动器在地面上固定的位置，第一列数据为促动器(通过下拉索)连接的主索节点的标号**(Index)**，后三列分别储存主索节点初始状态下**（X，Y，Z）**的坐标；注意到，由于促动器径向伸缩的范围有限，基准态和工作态下主索节点和促动器固定节点的距离之差不能超过0.6m;
- face.csv
   - 用于储存4300块三角形反射面相应的三个顶点，分为三列数据**（Index1，Index2，Index3）**，可以想象的是我们不难从这4300块反射面的顶点信息，得到6525根连接主索对应的顶点信息

​

我们可以使用以下代码将数据读入：
```python
import pandas as pd

# 数据读入
vertex_frame = pd.read_csv('./data/vertex.csv',header=0,sep=',')
ground_vertex_frame = pd.read_csv('./data/vertex_ground.csv',header=0,sep=',')
face_vertex_frame = pd.read_csv('./data/face.csv',header=0,sep=',')

# 可视化
vertex_frame.head()          # 主索节点坐标
ground_vertex_frame.head()   # 促动器坐标
face_vertex_frame.head()     # 反射面顶点
```
输出：

| 
- vertex_frame.head() 
 | 
- ground_vertex_frame.head()
 | 
- face_vertex_frame.head()
 |
| --- | --- | --- |
| ![frame_v.png](https://cdn.nlark.com/yuque/0/2021/png/22279603/1632884895959-c5c78c38-dc7b-477c-be3a-7084c9e7ffa9.png#clientId=u42b4c559-eb15-4&from=ui&height=125&id=ubd8e167e&margin=%5Bobject%20Object%5D&name=frame_v.png&originHeight=211&originWidth=394&originalType=binary&ratio=1&size=20273&status=done&style=none&taskId=u91b8e68c-6ae8-45ac-b6dd-be14b1c2a6c&width=234) | ![frame_g.png](https://cdn.nlark.com/yuque/0/2021/png/22279603/1632884944634-12d05324-27a3-4c48-87d7-86013836e9d5.png#clientId=u42b4c559-eb15-4&from=ui&height=125&id=u0562711e&margin=%5Bobject%20Object%5D&name=frame_g.png&originHeight=194&originWidth=367&originalType=binary&ratio=1&size=23125&status=done&style=none&taskId=ua16675e1-161b-4ab1-b583-096ba28aa7a&width=236) | ![frame_f.png](https://cdn.nlark.com/yuque/0/2021/png/22279603/1632884959190-f5909706-77cc-405a-a503-2a83f40071ee.png#clientId=u42b4c559-eb15-4&from=ui&height=125&id=u77d3838e&margin=%5Bobject%20Object%5D&name=frame_f.png&originHeight=206&originWidth=299&originalType=binary&ratio=1&size=11788&status=done&style=none&taskId=ubded0c77-9cbf-4de7-b638-b3dadef0d6e&width=181) |

       其中`vertex_frame`、`ground_vertex_frame`、`face_vertex_frame`为`pd.DataFrame`类，我们可以通过`DataFrame.head()`函数展示每张列表的前5条数据，从而对每张表内的数据获得直观的印象。
​

## 模型的建立
        为了使读者能够建立对这个工程问题更加直观的印象，在这里我将简单呈现更多有关于该工程的定量数据，同时附带一定的讨论。如果我们将主动反射面视作一个图（`Graph`），那么其上的2226个主索节点就对应着图的顶点（`Vertex`），6525根连接主索对应着图的边（`Edge`），4300块反射面则由各个连接主索作为边围成。
        在初始状态下，所有的主索节点都分布在一个半径为500m的基准球面上，我们认为此时主索节点都处在良好的工作状态，内部应力和形变量成正比。
​

4300块反射面的顶点信息，得到6525根连接主索对应的顶点信息
        
**可微分编程——jax实现**




















## 其他
### 可能可以作为图片来源的视频
[https://www.youtube.com/watch?v=4DANl-2SsPM&t=107s](https://www.youtube.com/watch?v=4DANl-2SsPM&t=107s)
[https://www.youtube.com/watch?v=k8yRDrl76d0&t=410s](https://www.youtube.com/watch?v=k8yRDrl76d0&t=410s)
​[https://www.youtube.com/watch?v=7SRV3rnULO0](https://www.youtube.com/watch?v=7SRV3rnULO0)
​

### 三张表的数值形式留备份

- vertex_frame.head() 
|  | Index | X | Y | Z |
| --- | --- | --- | --- | --- |
| 0 | A0 | 0.0000 | 0.0000 | -300.4000 |
| 1 | B1 | 6.1078 | 8.407 | -300.2202 |
| 2 | C1 | 9.8827 | -3.211 | -300.2202 |
| 3 | D1 | 0.0000 | -10.391 | -300.2202 |
| 4 | E1 | -9.8827 | -3.211 | -300.2202 |

- ground_vertex_frame.head()
|  | Index | Xg | Yg | Zg |
| --- | --- | --- | --- | --- |
| 0 | A0 | 0.0000 | 0.0000 | -304.7218 |
| 1 | B1 | 6.1935 | 8.5250 | -304.4318 |
| 2 | C1 | 10.0227 | -3.2560 | -304.4747 |
| 3 | D1 | 0.0000 | -10.538 | -304.4868 |
| 4 | E1 | -10.0214 | -3.2560 | -304.4337 |

- face_vertex_frame.head()
|  | Index1 | Index2 | Index3 |
| --- | --- | --- | --- |
| 0 | A0 | B1 | C1 |
| 1 | A0 | B1 | A1 |
| 2 | A0 | C1 | D1 |
| 3 | A0 | D1 | E1 |
| 4 | A0 | E1 | A1 |

​

​

​

​

​

