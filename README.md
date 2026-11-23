# MiniDuck 从零训练与部署仿真

沈佳乐 · 机器人学课程小组项目 · 第二个代表作品

项目在 MuJoCo 中验证一台双足机器鸭的步态、复杂地形控制与迷宫导航。基础步态通过 PPO 训练得到。部署仿真以通过基础验证的策略为起点，在控制层增加了坡度识别、步态配速、下坡恢复和稳定停机。机器鸭从封闭迷宫中心出发，根据模拟测距选路，依次通过 5°–10° 六段坡，穿过出口后完成 10 秒庆祝和 2 秒稳站。

## 完整仿真录像

[在 MiniDuck 项目网页播放](https://3535348582-prog.github.io/miniduck-simulation/) · [在 GitHub 查看或下载完整 MP4](docs/assets/miniduck-maze-recovery-1x.mp4)

录像为种子 `105391` 的整次原生仿真回放，**3 分 40.56 秒，1 倍速**。从中心起步直到出口庆祝与舞后稳站均保留。橙色表示上坡，黄色表示坡顶，青色表示下坡。录像的校验值见[评测数据](results/evaluation-summary.json)。

## 训练与部署流程

1. **基础步态。** 在课程环境中建立机器人观测、关节动作和物理场景，使用 PPO 分阶段学习平地直行与坡道步态，保留训练检查点和原生仿真评测结果。
2. **坡道控制。** 将冻结策略与前方地面探测、分段配速和姿态前视结合。接近陡坡末端时，恢复控制限制向前动量。控制器检测到稳定支撑后，逐步恢复平地配速。
3. **封闭迷宫。** 仿真测距更新局部通行信息。规划器比较候选前进和转向路径，遇到走不通的分支时重新选路。固定地图含一个出口与六段不同坡度的上下坡。
4. **终点验收。** 记录各坡段的足部接触与通过顺序，并检查越线、停稳、庆祝与舞后站立。原生执行记录可再次渲染为等速回放。

```mermaid
flowchart LR
    A[仿真测距与地面探测] --> B[局部选路与坡段识别]
    B --> C[速度 步频 航向]
    C --> D[步态策略与姿态前视]
    D --> E[MuJoCo 步进与观察]
    E --> A
    E --> F[越线 停稳 庆祝]
```

## 仿真成绩

| 评测 | 成绩 | 验收范围 |
| --- | --- | --- |
| 固定 10° 上下坡 | **100/100**，0 摔倒、0 碰撞 | 固定几何与摩擦，独立随机起点。通过样本的停稳用时中位数为 52.46 秒。 |
| 封闭迷宫 | **3/3**，完成代表回放的种子为 `105391` | 每次依次通过 5°–10° 六段坡并在出口庆祝，停稳用时中位数为 208.50 秒。 |

两组结果来自 MuJoCo 3.6.0 原生仿真。固定单坡和全程迷宫各自采用对应的控制配置与验收条件。[查看逐个迷宫试次与 10° 批次指标](results/evaluation-summary.json)

## 代码导航

| 入口 | 内容 |
| --- | --- |
| [center_exit_recovery.py](work/miniduck_fixed_maze/center_exit_recovery.py) | 最终完整迷宫控制器与下坡恢复条件 |
| [center_exit_adaptive.py](work/miniduck_fixed_maze/center_exit_adaptive.py) | 根据坡面、姿态和到达阶段切换控制 |
| [scan_planner.py](work/miniduck_fixed_maze/scan_planner.py) | 模拟测距与局部路径规划 |
| [slope_preview.py](work/miniduck_fixed_maze/slope_preview.py) | 姿态与支撑预测 |
| [map_center_exit.json](work/miniduck_fixed_maze/map_center_exit.json) | 实验场地、出口与 5°–10° 坡段定义 |
| [center_exit_test.py](work/miniduck_fixed_maze/center_exit_test.py) | 地形几何、完整试次与通过判据 |
| [render.py](work/miniduck_fixed_maze/render.py) | 从保存的姿态轨迹生成 1 倍速录像 |
| [run_sim.py](run_sim.py) | 本地检查、迷宫运行和录像入口 |

## 本地运行

下载仓库后，使用 Python 3.11 或更新版本检查公开文件。这个步骤只依赖 Python 标准库。

```bash
python run_sim.py check
```

### 从零训练试跑

在匹配的 Linux GPU 课堂环境中，可用下列命令复跑从随机初始化开始的 **2,048 万步 PPO 试验**。[训练配置](configs/ppo-probe-20m.json)记录了环境数、学习率调度、网络层和随机种子。先在本仓库根目录执行，再切换到已配置的课堂目录。

```bash
project_dir="$PWD"
cd /path/to/miniducktraining_classroom
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest OPENBLAS_NUM_THREADS=1
python miniduck.py train --output "$project_dir/runs/ppo-probe-20m" \
  --config "$project_dir/configs/ppo-probe-20m.json" --seed 20261028
```

试跑用于比较基础步态。完整迷宫录像使用后续筛选的冻结策略和下方的部署控制流程。

### 完整迷宫复跑

完整场地复跑使用经验证的 **Python 3.13**、[依赖版本](requirements.txt)和两类获得授权的本地输入。`--classroom` 指向完整的匹配课堂环境，含运行时策略加载器、场景 XML 及其引用的模型、网格和材质。`--runtime-assets` 指向含 `models/duck_initial/slope6_ppo_p3_128k.npz` 与 `models/duck_initial/accepted_6deg_final.json` 的本地目录。这里演示所用的策略、策略契约和课堂资产由获授权的项目交付环境提供。请按课堂环境的说明配置依赖，使用匹配的权重、接口和场景版本。

```bash
python -m pip install -r requirements.txt
python run_sim.py maze --classroom path/to/classroom --runtime-assets path/to/runtime-assets --out runs/maze --seeds 105391
python run_sim.py render --classroom path/to/classroom --runtime-assets path/to/runtime-assets --run runs/maze --seed 105391
```

重复运行时使用新的输出目录。`report.json` 保存每个试次的判定，`center_exit_105391.npz` 保存原生仿真姿态，`center_exit_105391_1x.mp4` 是生成的等速录像。视频渲染需要可用的 OpenGL 环境，可通过 `MINIDUCK_FONT` 指定支持中文的字体文件。

## 来源

项目参考 Open Duck 的训练与运行时设计，仿真使用 MuJoCo。课堂场景、模型和第三方工具各自沿用原来源许可。仓库公开展示项目控制代码、迷宫参数、评测摘要和最终录像。[Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground) · [Open Duck Mini Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) · [MuJoCo](https://github.com/google-deepmind/mujoco)
