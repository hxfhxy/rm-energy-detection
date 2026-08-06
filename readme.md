# 能量单元位姿识别与优化 - 程序设计报告

## 一、 整体流程设计

主要包含以下五个处理阶段：

1. **2D 掩码与目标检测 (YOLO-Seg)**：读取 RGB 与 Depth 图像帧，利用 OpenVINO 加载 YOLO-Seg 深度学习网络，在 UV-RGB 重投影图上获取能量单元的 2D 包围框（ROI）、中心点、类别置信度以及分割掩码（Mask）。
2. **深度图重投影与点云提纯**：利用深度相机内参将 2D 图像掩码与深度图反投影至 3D 空间，依次进行体素下采样、统计学离群点剔除与欧式聚类，提取纯净的目标点云。
3. **PCA 主轴提取与极性定向**：对滤波后的点云执行 PCA，提取长轴方向；利用能量单元"大头/小头"的半径差异特征（底部半径大、顶部半径小），强制令物体 +Z 轴指向小头（底部），一次性解决 180° 极性歧义。
4. **多候选 Roll 生成与 Trimmed ICP 精炼**：根据圆柱体的旋转对称性，绕 Z 轴均匀生成多组离散 Roll 角与双极性候选；利用相机投影 Y 坐标约束快速过滤物理不合理的候选；将 CAD 模型点云与场景点云送入 Trimmed ICP 进行非线性对准，按 trimmed RMSE 与 correspondence ratio 选出最优 4×4 变换矩阵。
5. **XY 轴正交化固定与可视化**：对 ICP 输出的旋转矩阵进行投影正交化，强制 X 轴水平朝右、Y 轴由右手定则确定，消除圆柱对称性导致的 Roll 角漂移；最终在 OSD 界面与终端实时渲染 6D 姿态与解算耗时。

---

## 二、 函数结构组织

### 1. 前端检测与推理

* **主要职责**：负责从 UV-RGB 重投影图像中推理目标，解析 YOLO-Seg 输出结构，生成 2D 包围框与分割 Mask。
* **核心函数**：
  * `EnergyUnitDetector::runInference()`：OpenVINO 推理入口，解析 Det Tensor 与 Proto Mask Tensor。
  * `EnergyUnitDetector::decodeMask()`：解算 32 维掩码系数并进行 Sigmoid 激活、ROI 裁剪与膨胀恢复，输出二值掩码 Mat。
  * `EnergyUnitDetector::getDepth()`：在 2D 中心点附近扩大窗口采集深度中位数。

### 2. 空间点云预处理与滤波

* **主要职责**：从深度图抽取 3D 点云，经过多级空间滤波剔除背景与高光噪点。
* **核心函数**：
  * `PoseEstimator::extractPointCloud()`：根据 Mask 提取有效深度点，通过掩码内深度中值门控（±10 cm）剔除背景离群点。
  * `PoseEstimator::voxelDownsample()`：空间体素网格下采样，均匀点云密度，提升后续计算效率。
  * `PoseEstimator::statisticalOutlierRemoval()`：统计学滤波（SOR），计算每个点到近邻的平均距离，剔除高噪声飞点。
  * `PoseEstimator::euclideanClusterSelect()`：欧式聚类生长，自动选取点数最多的最大簇，隔离背景墙面与邻近物体。

### 3. 主轴几何解算与极性定向

* **主要职责**：求解圆柱体物体的 3D 中轴线方向并消除 180° 极性翻转。
* **核心函数**：
  * `PoseEstimator::computePCALongAxis()`：计算协方差矩阵并求解特征值，输出最大特征值对应的特征向量作为初级主轴。
  * **大头/小头半径判别**：沿 PCA 长轴将点云分为上下两段，分别计算到轴线的平均垂直距离；底部（大头）半径大、顶部（小头）半径小，据此强制令 +Z 指向小头，锁定极性。

### 4. 候选姿态生成与 ICP 优化器

* **主要职责**：构建多组初始 Roll 候选，通过 Trimmed ICP 最小二乘对准，输出高精度刚体变换。
* **核心函数**：
  * `PoseEstimator::buildModelCloud()`：根据能量单元参数（半径 47.5 mm，高度 150 mm）离散化生成圆柱体 3D 模型点云（侧面 + 顶底圆盘）。
  * `PoseEstimator::generateInitialCandidates()`：以 PCA 质心为平移初值，构造 2 极性 × N 个离散 Roll 角 = 2N 个初始位姿候选。
  * **投影 Y 约束筛选**：将候选姿态下的模型顶/底中心重投影到图像平面，仅保留"顶部像素 Y 坐标 < 底部像素 Y 坐标"（物理上顶部更靠近图像上方）的合法候选。
  * `PoseEstimator::trimmedICP()`：基于 KD-Tree 最近邻搜索与 SVD 求解，利用 Trimmed 比例过滤高残差离群对应点，迭代收敛后输出最优 T、trimmed RMSE 与 correspondence ratio。

### 5. 节点调度与可视化层

* **主要职责**：ROS2 节点调度、RGB/Depth 时间戳同步、双重触发锁与 OpenCV 渲染。
* **核心函数**：
  * `EnergyUnitNode::reprojectDepthToRGB()`：将深度图反投影至 3D 相机坐标系，再投影到 RGB 图像平面，生成与深度图尺寸对齐的 UV-RGB 彩色图（当前默认深度与彩色外参 R=I, t=0，建议后续接入真实外参或改用相机驱动对齐深度话题）。
  * `EnergyUnitNode::TryProcessFrame()`：时间戳容差匹配（< 10 ms），防止掉帧与重复触发。
  * `EnergyUnitNode::processFrame()`：主流程入口，调度检测、姿态估计、XY 轴正交化锁定与 OpenCV 渲染。

---

## 三、 具体重要算法

### 1. 2D 掩码的 3D 点云提取与过滤

* **输入**：OpenVINO 解码出的 `det.mask`、深度图 `depth`、相机内参。
* **输出**：3D 空间场景点云。
* **步骤**：
  1. 遍历 2D 掩码 ROI 内的有效像素 ；
  2. 统计掩码内所有有效深度的中位数，保留满足 |d - d_median| <= 10 的点，剔除背景与离群深度；
  3. 利用针孔相机模型反投影为 3D 点

### 2. PCA 长轴提取与大/小头极性判定

* **输入**：滤波后的纯净点云。
* **输出**：带有明确方向的物体 Z 轴（长轴）与质心。
* **步骤**：
  1. 计算点云质心与协方差矩阵 ；
  2. 特征值分解，取最大特征值对应的特征向量作为长轴方向（此时无符号）；
  3. 将点云沿 向量V 投影，以中位数投影值为界分为两段；
  4. 分别计算两段点云到轴线的平均垂直距离（平均半径）。底部（大头）半径大、顶部（小头）半径小；
  5. 若上半段平均半径大于下半段，则将 向量v 反向，确保 Z 始终指向顶部。


### 3. 多候选 Roll 生成与投影 Y 约束
* **输入**：PCA 质心 、定向后的长轴 、离散采样数 。
* **输出**：经物理约束筛选后的合法初始位姿候选集合。
* **步骤**：
  1. 构造基础旋转 R_base，使物体坐标系 Z 轴与 z 对齐；
  2. 绕 z 轴以 $\Delta\theta = 2\pi / N$ 为步长采样，生成 N 个 Roll 角；
  3. 同时考虑长轴的两个极性（已在大/小头判定中确定主极性，此处保留双极性以覆盖歧义），共得到 2N 个候选 T_i；
  4. 对每个候选，将模型局部坐标下的顶部 (0,0,h/2) 与底部 (0,0,-h/2) 变换到相机坐标系并投影到图像平面
  5. 仅保留满足 v_top < v_bot（图像坐标系 Y 轴向下，顶部在图像上方则 Y 值更小）的候选，剔除倒置姿态。

### 4. Trimmed ICP 点云对准

* **输入**：CAD 理论模型点云 M、滤波后的场景点云 S、初始姿态矩阵 T_0。
* **输出**：精炼后的 4×4 刚体变换矩阵 T、trimmed RMSE、correspondence ratio。
* **步骤**：
  1. **最近邻搜索**：利用 KD-Tree 为当前变换后的模型点 M' 寻找场景点云 S 中的最近邻，仅保留距离小于 `max_dist` 的匹配对；
  2. **Trimmed 截断**：计算所有匹配对的欧氏距离残差并升序排列，保留前 r 比例（默认 75%）残差最小的点对，强制抛弃高噪离群点；
  3. **SVD 刚体变换求解**：对保留的点对计算去质心协方差矩阵 H，进行 SVD 分解 ；
  4. 求解最优旋转 ，平移；
  5. 更新累计变换，迭代 5 次后输出最终 RMSE 与匹配率。

### 5. XY 轴投影正交化固定

* **输入**：ICP 输出的旋转矩阵 R。
* **输出**：X/Y 轴被唯一确定的正交旋转矩阵 R_fixed。
* **步骤**：
  1. 保持 Z 轴不变：z = R_2；
  2. 将世界 X 轴 (1,0,0)投影到垂直于 z 的平面：
     若 |x| 过小（z 接近水平），则到 e_y 投影；
  3. 归一化 x，并由右手定则 y = z.x 得到 y；
  4. 重组 R_fixed = [xyz]，消除圆柱绕轴旋转对称性带来的 Roll 漂移。

---

## 四、 数据组织

### 1. 2D 目标检测结构 (`Detection`)

```cpp
struct Detection {
    cv::RotatedRect bbox;             // 2D 旋转包围框
    cv::Rect bbox_axis_aligned;       // 2D 正外接矩形 (ROI)
    float confidence;                 // YOLO 分类置信度
    cv::Point2f center;               // 2D 图像框中心点
    float depth_value;                // 中心区域中值深度 (mm)
    int label;                        // 类别标签 (如 0: 能量单元)
    cv::Mat mask;                     // 单通道 8UC1 二值掩码
};
```

### 2. 估计器配置参数结构 (`EstimatorConfig`)

```cpp
struct EstimatorConfig {
    // 相机内参（深度相机）
    double fx = 424.49407958984375;
    double fy = 424.49407958984375;
    double cx = 422.3330383300781;
    double cy = 241.14089965820312;

    // 圆柱物理尺寸（m）
    double cylinder_radius = 0.0475;   // 半径 47.5 mm
    double cylinder_height = 0.150;    // 高度 150 mm

    // 点云滤波参数
    double voxel_leaf_size = 0.01;     // 体素下采样分辨率 (1 cm)
    double cluster_tolerance = 0.015;  // 欧式聚类距离阈值 (1.5 cm)
    int cluster_min_size = 50;         // 最小簇点数
    int stat_outlier_k = 20;           // 统计滤波近邻数
    double stat_std_mul = 1.0;         // 标准差倍数

    // ICP 参数
    int icp_max_iter = 5;                           // ICP 迭代次数
    double icp_max_correspondence_dist = 0.05;      // 对应点最大距离 (m)
    double trimmed_ratio = 0.75;                    // Trimmed 保留比例
    int num_roll_candidates = 4;                     // 绕轴 Roll 离散采样数

    // 调试
    bool verbose = false;
};
```

### 3. 6D 位姿输出结构 (`PoseResult`)

```cpp
struct PoseResult {
    bool valid = false;                // 解算有效标志
    int label = 0;                     // 目标类别
    double confidence = 1.0;           // 位姿解算置信度

    Eigen::Vector3d translation;       // 3D 平移向量 (X, Y, Z in meters)
    Eigen::Quaterniond quaternion;     // 姿态四元数
    Eigen::Vector3d axis_direction;    // 3D 轴线向量 (Z 轴方向)
    Eigen::Vector3d euler_angles;      // 欧拉角 (RPY in degrees)
    Eigen::Matrix4d T_cam_obj;         // 相机到目标的 4×4 齐次变换矩阵
};
```

---

## 五、 遇到的难点及解决思路

1. 深度与彩色的对齐

在提取点云时，我发现常规的把深度图插值到彩色图上的做法，会产生大量虚假深度。所以，我换了个思路，遍历深度图，去彩色图里获取颜色，把彩色重投影到深度的坐标系下（UV-RGB）。虽然这样可能会在能量单元的盲区留下几个空同，但起码把 YOLO 检测和深度图的物理视差给对齐了。

2. PCA 找轴与“大头小头”判定

用 PCA 找圆柱中轴线是大家都会做的，但 PCA 有个缺陷：算出的轴有 180 度极性翻转的歧义。如果单纯依赖图像 Y 坐标来做判断，能量单元一倾斜就会跳变。所以我就想到了利用物理几何特征——能量单元底部安装座（小头）半径小，顶部（大头）半径大。把点云沿轴切两半，算一下平均距离，强行让 Z 轴指向小头。用这种方法定出来的方向稳定。

3. Trimmed ICP

在此之前，ICP 对齐方面我也尝试了普通的 ICP 算法，或者在外面套一层多帧时序滤波。多帧滤波我觉得会有明显的滞后性和污染性，用了之后数据会有明显的拖影，尽管可以调参来减小，但还是会有轻微滞后
普通 ICP 一遇到噪点就容易被拉偏，所以在 AI 的帮助下，我搞了个 Trimmed ICP。找最近邻的时候用 KD-Tree 查字典加速，算完距离后，直接去除误差最大的 25% 垃圾点，只用最靠谱的 75% 跑 SVD 算旋转矩阵。


## 六、 总结

这道题我感觉不是很熟悉，因为第一次接触，所以说实话挺手足无措的，就是没有很通畅的思路，基本都是在和ai一边学习一边完成，有点惭愧，而且一开始我是使用了m3t的，不过后来看到群里老登说不建议使用m3t就被劝退了，然后我直接全部重构我的代码，不过真的没时间啊，我下定决心重构的时候，只剩3天左右了，所以最后这几天真的急哭了，进阶的机械臂坐标转换实在没时间做了

因为实在没有时间细细研究这道考核题，所以我就说一下我学习到了什么，

体素网格下采样：通过构建三维空间网格，将原本密集的点云离散化，每个体素块内只保留一个点，这个算法可以很好降低计算的空间复杂度
统计学滤波： 计算每个点到周围 k 个邻居的平均距离，通过假设距离符合高斯分布，利用均值和标准差去除掉那些距离过远的噪声
K-D树：建立一个空间索引，一层一层地切开分块。比如第一层按 X 轴把点分左右，第二层按 Y 轴分上下，第三层按 Z 轴分前后，建立一棵树状结构，这样子找最近邻点的时候，就不需要一个个遍历，而是可以有方向的搜寻，可以很好提升算法效率
icp：我认为icp其实本质上就是“不断试错、不断靠近”的迭代优化，就是把模型的 3D 点云不断遍历匹配图像的点云，找到最贴合的状态
Trimmed ICP ：我觉得和icp的区别就在于，他会算出所有点对之间的距离误差，然后淘汰掉误差大的

这些就是我觉得学习到比较有用的算法部分


最后我的流程就是首先通过彩色到深度的反向映射对齐两种图像，接着利用 YOLO-Seg 掩码提取目标，并用空间门控、体素、SOR 与欧式聚类将点云筛选；随后通过 PCA 结合大头小头的半径特征精准定出主轴与正反极性，以此生成多组旋转初值喂给 Trimmed ICP，靠 KD-Tree 检索去除 25% 边缘噪点实现 3D 模型的贴合，最后提取真实表皮的物理锚点重构正交坐标系
---

## 七、 编译与运行

```bash
# 编译
cd ~/能量单元位姿识别
colcon build --packages-select energy_unit_pose

# 启动
source install/setup.bash
export LD_LIBRARY_PATH=~/.local/lib/python3.10/site-packages/openvino/libs:$LD_LIBRARY_PATH
ros2 launch energy_unit_pose demo.launch.py

# 另一个终端播放数据
ros2 bag play ~/能量单元位姿识别/text3/my_d435i_bag_0.db3 --rate 0.5
```