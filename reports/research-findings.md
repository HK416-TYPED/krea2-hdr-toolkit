# Krea 2 → 真实场景线性 HDR(EXR)研究发现报告

> 对应任务书:`2026-07-07-krea2-hdr-t2i-research-brief.md`
> 完成日期:2026-07-08
> 方法:6 路并行深度检索(每个探索区域一路)+ 第一手核查关键事实(Krea 2 / Qwen Image VAE 源码、X2HDR、Sumit、S2R-HDR)。所有 arXiv ID 均已逐一验证存在;与任务书原文有出入处会明确标注。

---

## 0. 执行摘要:最重要的三个判断(任务书说这类判断价值最高,故置顶)

### 判断一:你们的"路线 A"是对的、可行的、有多篇独立证据支撑——但它的产物是**显示级 HDR(~10–14 stop / 4000 nit),不是 IBL 级的场景线性辐射**。

- **X2HDR(arXiv 2602.04814)就是你们计划的那套架构**(PU21 编码 + 冻结 VAE + 只训去噪器 LoRA)。它已被我第一手核实存在且结论成立:LDR 预训练 VAE 对 **PU21 编码后的 HDR** 的往返保真度(JOD 9.44)几乎等同于对真 LDR(9.86),而**直接喂线性 RGB 会崩到 8.54**。这是路线 A 最强的单点证据。
- **但**同一篇 X2HDR 自己承认:它把一切**重标定到峰值 4000 cd/m²(消费级 HDR 电视目标)**,并出现"不可信的幻觉""夸张的全局对比""极端高光被大面积裁掉且不可恢复"。**4000 nit / 14 stop 是显示 HDR**。而 IBL 用的太阳盘面在漫反射白之上约 10⁴–10⁵ 倍(≈17–20+ stop),且必须**绝对定标**才能让渲染器算对光。
- 结论:**"小 LoRA + 感知编码 + 冻结 VAE"作为第一步 PoC 是正确、最省的实验,预期能成功——成功到 ~10–14 stop 的"看起来对"的 HDR。但它在结构上不可能成为最终交付真·场景线性、覆盖太阳量级 EXR 的终点方案。**

### 判断二:天花板是**算术性的,不是调参能突破的**;瓶颈在**编码器 + 潜空间的精度预算**,不在 LoRA、也不在解码器的那个 clamp。

- 我第一手读了 diffusers 的 `autoencoder_kl_qwenimage.py`:Qwen Image VAE 的解码器输出**没有** tanh/sigmoid 有界激活,唯一的边界是 `_decode` 里一行 `torch.clamp(out, -1, 1)` 加上输入预处理把 `[0,1]→[-1,1]`。**去掉这个 clamp 是一行改动、不碰任何训练权重。**(任务书阶段 1 说的"VAE 输出经 tanh/sigmoid 有界"对 Qwen VAE 不准确——这点很重要,因为它意味着"解码器根本不是瓶颈"。)
- 真正的瓶颈在**编码侧**:冻结编码器只在 `[-1,1]` 显示参考数据上训练过,对 >1 的高光**没有分配任何容量**;直接喂线性 >1 = 越界/裁剪 = 不可靠。**你无法解码出编码器从没编码进去的信息。**
- 而当你先用 log/PQ 把范围**压进** `[-1,1]` 后,冻结编码器看到的是"分布内"的数据,能忠实往返——展开回线性发生在 VAE **之外**。这就是路线 A 能用完全冻结 VAE 工作的原因。
- 精度上限是算术:Qwen VAE 8× 空间压缩 + 约 8 bit 有效精度。**⚠️ CPU 实测修正(见 §0.5)**:早先"约 85 码 → 4–6 stop"的算术估计**过于简化**——log 曲线在均匀 8-bit 下其实给出均匀 ~17 码/stop、天花板以下无 banding 墙;真正的限制是 VAE 的**非均匀**精度(码值偏向训练过的显示参考中间调),高光在逼近 V→0.9 时才劣化。净效果仍成立:**单平面干净承载 ~10–11 stop,但太阳(比中调高 ~19 stop)必然溢出曲线天花板而裁剪**。这解释了 X2HDR 停在 14 stop、冻结 VAE 逐帧包围融合(2604.21008)只到 ~4.5 stop、以及真实 Poly Haven HDRI 上太阳带 97–99% 误差(§0.5)。

### 判断三:要拿到**真·覆盖太阳的场景线性 EXR**,业界收敛到三条路——都不是"纯冻结 VAE 单平面":

1. **改造/扩容 VAE**(LuxDiT 的双 tonemap 表示、Sumit 的 16→32 双头)——把 HDR 信息真正编进潜空间;
2. **分解成多平面**(曝光包围栈 / 增益图),让每个平面都待在 VAE 的舒适区,极端范围放在**单独的平面**里;
3. **对天空用参数化太阳+天空模型**(Lalonde-Matthews / Hošek-Wilkie / Prague Sky / LM-GAN)——这是**唯一能让太阳盘面按构造就辐射正确**的一类方法。

> **给技术顾问的一句话总纲**:第一步就按路线 A 做(它便宜、能验证链路、能出漂亮的显示级 HDR),但**从第一天起就把它当"第一步"而不是"终点"来架构**——数据用真场景线性、编码用相机 log、保存链路杜绝隐式 clamp,并**用"渲染探针"做辐射级验证而不是只看感知指标**。真要 IBL 级,第二步必然走"扩容 VAE 或分解多平面",天空再叠一个参数化太阳。

---

## 关键事实核查表(第一手,修正任务书若干处)

| 事实 | 核查结果 | 对任务书的修正 |
|---|---|---|
| X2HDR = arXiv 2602.04814 | ✅ 真实(2026-02-04),核心结论成立 | 无 |
| Krea 2 架构 | ✅ 单流 MMDiT 12.9B / 28 block / width 6144 / GQA+gated sigmoid / SwiGLU 4× / 3D RoPE;Qwen3-VL-4B-Instruct;Qwen Image VAE;RAW+Turbo(8 步)。**官方 diffusers `krea2` pipeline 已存在**,kohya-ss musubi-tuner 有 krea2 训练文档 | 无 |
| Qwen Image VAE 解码器"有界激活" | ⚠️ **不准确**:无 tanh/sigmoid,只有一行 `clamp(-1,1)`。它是 Wan 系因果 3D VAE(z_dim 16, dim_mult [1,2,4,4]) | 修正阶段 1"VAE 输出经 tanh/sigmoid 有界"的表述 |
| Sumit Chatterjee 方法 | ✅ 真实。Phase1 LogC4+LoRA(峰值 200–300,-2EV banding);Phase2 全量微调 VAE(编码器解冻)+ 可学习逐通道归一化 + 16→32 双头(tonemap+linear 共享潜码)+ rank-64 MMDiT LoRA。~9000 EXR,单卡 5090 约 26h,峰值 >500 | 无 |
| Felldude VAE | ✅ 确认仅解码器、编码器冻结、全局曝光不变 → 锐化器非 HDR。已排除正确 | 无 |
| S2R-HDR | ✅ 24000 帧 = 1000 序列 ×24,1920×1080,**透视非全景**,UE5 场景线性 EXR,ICLR 2026。**未定标太阳(帧内 DR 仅 ~12.8 stop)、无 caption、有效场景 ~1000**。S2R-HDR-2 是同一发布的后半,非 v2 | 数据集可用但**只作场景线性"卫生"来源,不作 IBL/太阳骨干** |
| 新出现且相关 | Qwen-Image-VAE-2.0(2605.13565)、LumaFlux(2604.02787)、LatentHDR(2605.11115)、曝光包围线性生成(2604.21008)、GMODiff(2512.16357)等 | 任务书未覆盖,下文纳入 |

---

## 0.5 CPU 实测验证(2026-07-08,非 GPU 部分已落地)

> 代码与可复现结果在 `hdr_cpu_validation/`(`python3 run_all.py`)。这部分把上面的判断从"文献综述"变成了"在真实场景线性数据上跑过的数字"。**未用任何模型/GPU**——VAE 用"均匀/非均匀量化"两个 CPU 代理近似,其余(编码往返、EXR 读写、数据集分析)是真实计算。

**已验证的结论:**
1. **线性↔编码的映射与还原数值上精确**:8 条曲线(Log3G10/LogC3/LogC4/PU21/PQ/asinh/Log-Gamma/µ-law)在 −10EV..天花板范围内往返误差 ~1e-13(float64),可安全放进 ComfyUI 图。
2. **中灰落点符合理论**:code@0.18 = Log3G10 0.333 / LogC3 0.391 / LogC4 0.278 / PU21 0.359 / PQ 0.348(都贴近 sRGB ~0.46);µ-law 0.015、Log-Gamma 0.809 严重错位 → 实测确认是坏选择。
3. **⚠️ 对判断二的重要修正**:在**均匀** 8-bit VAE 代理下,log 曲线给出**均匀 ~17 码/stop、天花板以下无 banding 墙**——所以"~85 码 → 4–6 stop 算术天花板"这个说法**过于简化、被实测推翻**。真正的限制来自 VAE 的**非均匀**精度(它把码值花在训练过的显示参考中间调上)。非均匀代理显示:高光在逼近 V→0.9 时才开始劣化,且**头顶空间更大的曲线(LogC4/asinh/Log3G10)更晚 banding**——这定量验证了"峰值压在 V≈0.9 以下"和"优先高头顶空间曲线";LogC3 天花板低(55)最早 banding。
4. **EXR 链路(区域 5)端到端验证**:32-bit float ZIP/PIZ/ZIPS **精确**保住 4000 辐射的太阳;DWAA 有损偏移 ~1.6%;**half 在 >65504 溢出成 +Inf**。两个沉默杀手复现:clamp→[0,1] 把 4000→1.0;隐式 sRGB OETF 把 4000→33.4。`--stats` 二分能正确区分"保存被裁"与"模型没生成 HDR"。
5. **真实数据确认核心论断(Poly Haven `kloofendal_43d_clear`)**:median 0.178(≈0.18 锚点✓),**max 111,253、DR 31.6 stop、含真太阳**。单平面 8-bit 代理还原:阴影/中调/高光/亮部各带**误差 <1%**(到 ~500),但**太阳带(>500)误差 97–99%**——因为它溢出每条曲线的天花板(仅 0.002% 像素,但那就是太阳)。**⇒ 单平面编码干净地承载 ~10–11 stop,但太阳本身必然裁剪 → 太阳量级需多平面**,与判断三一致。
6. **数据集角色实测确认**:Poly Haven kloofendal 峰值 111,253×(真太阳→IBL 骨干);**S2R-HDR 样本(scene_0_FM/0000)峰值仅 21×、无太阳** → 场景线性**卫生源**而非 IBL/太阳骨干,与区域 3 一致。

**唯一剩下的 GPU 测量**:把上面的"量化代理"换成真·Qwen VAE 的 encode→decode 往返(区域 1 的"决定性往返实验",约 1 GPU·小时),即可在训练前敲定曲线选择。→ **已完成,见 §0.6。**

---

## 0.6 真·Qwen VAE 往返实测(2026-07-08,GPU,决定性往返实验已完成)

> 用**真·Qwen Image VAE**(`Qwen/Qwen-Image` 的 vae 子文件夹,z_dim=16,与 Krea 2 同一个 VAE;实测确认解码器唯一边界是 `torch.clamp(out,-1,1)`、无 tanh/sigmoid)替换掉 §0.5 的量化代理,在真实 HDR 上对每条曲线做 encode→[0,1]→VAE encode/decode→inverse 往返。代码 `hdr_cpu_validation/test_vae_roundtrip.py`,结果 `out/vae_roundtrip_results.json` + 可视化 `out/vae_roundtrip_kloofendal_LogC4.png`。

**新增/确认的结论:**
1. **X2HDR 前提在真 VAE 上被强力证实**:感知编码的 HDR 往返码域 PSNR ~35–38 dB、中+高光带误差 ~2%;而**朴素线性(scaled-linear)直接喂 VAE → 1–50 带误差 7211%**。这就是"必须走感知编码"的真·VAE、真·数据铁证。
2. **真 VAE 上的曲线排名(码域 PSNR,kloofendal)**:LogC4 38.2 > PU21 37.7 > PQ 37.1 > Log3G10 36.9 > asinh 36.1 > LogC3 35.4。**LogC4 略胜 Log3G10**,且亮部(50–500)带最好;**LogC3 裁剪(亮部误差 44.9%)** → 真 VAE 确认其低天花板的问题。**→ 据此把首选起点从 Log3G10 微调为 LogC4(Log3G10 作 on-manifold 备选)。**
3. **真 VAE 两端都劣化**(量化代理漏掉的发现):深阴影/暗部高频纹理相对误差 ~22–24%(8× 压缩丢细节),太阳仍裁剪(97–99%)。干净窗口是中+高光(0.05–50)~2%。可视化显示误差集中在前景阴影草地/岩石,天空与中调干净,太阳裁剪。
4. **太阳在真模型上仍然裁剪** → 单平面极限用真 VAE 再次确认;太阳量级需多平面(区域 1/2/6)。

**净判断(用真 VAE 落地后)**:冻结 VAE + 感知编码路线在真 VAE 上端到端验证为**显示级 HDR 可用**(中+高光 ~2%、~10–11 可用 stop),阴影偏软、太阳裁剪——与"单平面非 IBL 级"的判断三完全一致。训练前的曲线选择已敲定:**LogC4 主选,Log3G10 备选**。

---

## 探索区域 1:让潜码携带扩展动态范围的"编码空间"

### 发现的方案全景

1. **PU21** —— X2HDR 基线,感知均匀,往返最佳但可学性一般。
2. **PQ / ST 2084** —— 头顶空间最大(10000 nit),但中灰位置偏亮、off-manifold、极端高光精度最差。
3. **ARRI LogC3 / LogC4** —— 相机 log 主力;LumiVid / LumiPic / LTX-2 / Sumit 都收敛到这里。
4. **RED Log3G10** —— ⭐ **首推**:on-manifold 的同时头顶空间最大、负值稳定,HDR-diffusion 界尚无人用过。
5. **DiffHDR Log-Gamma** —— 可调 log+gamma 映到有界 [0,1]。
6. **asinh / Lupton 拉伸(天文)** —— ⭐ **推荐第二实验**:近黑线性 + 顶部对数 + 单一柔度参数 + 闭式反变换。
7. **μ-law / A-law companding(音频)** —— 数学上等价于 log-companding,概念佐证,无额外收益。
8. **ACES / Reinhard / Uncharted2 / AgX-sigmoid** —— 游戏引擎 tonemapper,**多对一、裁剪去饱和、反变换=猜**,对线性 EXR 目标基本错误。
9. **Khronos PBR Neutral** —— 唯一解析可逆的 filmic,但头顶空间小、去饱和丢通道比。
10. **增益图 / 双表示(ISO 21496-1 / Ultra HDR)** —— ⭐ **通往太阳量级的唯一干净单平面外方案**:SDR 底图 + log 增益图。
11. **LatentHDR 潜到潜曝光解耦** —— 生成基础潜码 → 确定性映射出曝光栈 → log 域合并出真辐射。
12. **可学习 / 空间自适应编码** —— LumaFlux 单调 RQS 样条 tone-field 解码头(冻结 VAE 之外)、空间变化曝光场、局部 Laplacian 高频残差。

### 方案详情(节选最关键的三条)

#### 方案 1.4:RED Log3G10 编码 + 冻结 VAE + 去噪器 LoRA(首推起点)

**一句话原理**:用一条"18% 灰落在 1/3、灰之上 10 stop 映进 [0,1]、过零处有线性段"的相机 log 曲线,把场景线性压进冻结 VAE 的舒适区,输出后解析反变换回线性。

**灵感来源**:RED IPP2 / REDWideGamut(电影相机);与 LumiVid(2604.11788,LogC3+冻结 VAE)、Sumit Phase1(LogC4)同源但曲线更优。

**具体实现步骤**:
1. 前向(逐通道):`V = 0.224282·log10(155.975327·L + 1)`(L≥0);`V = L/15.1927 − 0.01`(L<0)。
2. 反向:`L = (10^(V/0.224282) − 1)/155.975327`。
3. 数据集:所有 EXR → Log3G10 编码 → VAE 常规预处理(现在每张图都"分布内")。
4. **只在 Krea 2 RAW 上训 LoRA**(rank 16–64,attn Q/K/V/O),VAE / 文本编码器 / patch-embed / final-layer 全冻结。
5. 推理:DiT → VAE decode(冻结)→ Log3G10 反变换 → 写场景线性 EXR + OCIO。

**涉及组件**:编码/反编码变换(两个纯函数节点)+ 去噪器 LoRA。VAE、潜空间维度、Turbo 蒸馏路径全不动。

**与现有系统对接**:
- 改哪:只加两个变换节点 + 一个 LoRA。
- 输入变:训练前把 EXR 过 Log3G10。
- 输出变:decode 后插反变换节点,再走 EXR 保存(见区域 5);最终张量是**真线性值**。
- **对 RAW→Turbo 迁移**:**最优,完全不破坏**。潜空间形状不变(只是数据流形变了),RAW 训的 log 空间 LoRA 应用到 Turbo 与任何风格 LoRA 一样——这是官方支持的路径。**这是路线 A 相对方案 B 最重要的优势。**

**难度评估**:**低**。无架构改动,一个 LoRA,确定性前后变换。一个周末能跑通。

**预期效果**:**真实** >1 辐射(反变换后)。头顶空间 ~+10 stop over gray(峰值几百 × gray),高于 LogC3。**EXR 意义上的"太阳量级"(峰值几百到低千)可达;字面太阳盘面(10⁴–10⁹)不可达——但没有任何单平面方法可达,EXR 管线也不需要它。**

**先例/参考**:https://antlerpost.com/colour-spaces/Log3G10.html ;LumiVid https://arxiv.org/abs/2604.11788 ;Sumit https://sumitc.com/work/hdr-generation

**风险与不确定性**:高光 banding(顶部码值稀疏,Sumit 在 -2EV 见到);中灰略暗于 sRGB;**务必先在你自己的 Qwen VAE 上做往返测试**(见下)确认 1/3-gray 锚点的表现再投入训练。

#### 方案 1.6:asinh / Lupton 拉伸(天文,推荐第二实验)

**一句话原理**:`asinh` 对小值线性(保住阴影/噪声底,甚至负值)、对大值对数(压尾),一个柔度参数 β 直接权衡阴影线性度与高光压缩。

**灵感来源**:Lupton, Gunn & Szalay 1999(SDSS "asinh magnitudes"),天文标准拉伸。

**具体实现步骤**:前向 `V = asinh(L/β)/asinh(Lmax/β)`;反向 `L = β·sinh(V·asinh(Lmax/β))`。β 取 0.01–0.05(近阴影尺度),Lmax = 固定峰值锚点。

**与现有系统对接**:即插即用变换,顶部像 log 故继承 LumiVid 式潜对齐,但近黑更优雅——直接缓解 Sumit 的高光 banding(banding 本质是精度分配问题)。**不破坏 RAW→Turbo 迁移。**

**难度评估**:低(实现)/ 中(调 β)。HDR-diffusion 界无人用过 → 属新贡献。

**预期效果**:真实,达 Lmax;顶部精度上限与任何 log 相同(近黑更好)。

**风险**:管线不熟悉;β 与 Lmax 必须**全局固定**以保 EXR 绝对性。

#### 方案 1.10:增益图 / 双平面(ISO 21496-1)——通往太阳量级的唯一干净路径

**一句话原理**:不把全部动态范围硬塞进一个平面;存一张**显示参考 SDR 底图**(完美 VAE 输入)+ 一张 **log 增益图** `gain = log2(HDR/SDR)`,重建 `HDR = SDR·2^gain`,增益平面原则上无界。

**灵感来源**:ISO 21496-1:2025、Google Ultra HDR、GM-Diffusion(ICCV 2025)、GMODiff(2512.16357)、Gain-MLP(2503.11883)。

**具体实现步骤**:底图 = tonemap(HDR) in sRGB;`gain = clamp(log2((HDRlum+ε)/(SDRlum+ε)), gmin, gmax)`,增益图归一到 [0,1] 存元数据(gmin,gmax);逐像素重建。生成侧两种做法:(a) 用**未改动的模型**生成 SDR 底图 + 一个**第二 LoRA / 第二潜平面**生成增益图(增益图平滑、低频 → VAE 很容易);(b) 底图与增益图各自过一次 VAE。

**与现有系统对接**:底图保持完全分布内 → 往返极佳;增益平面承载任意头顶空间 → **可达太阳量级**(仅受 gmax 与增益平面精度限制)。SDR 分支的 RAW→Turbo 迁移保持干净。

**难度评估**:中–高(双平面管线、两次 decode、对齐)。但这是对"太阳量级"要求**唯一诚实**的单平面外答案。

**预期效果**:真实且(至 gmax)无界。**重要局限**:ISO 增益图是**显示端构造**(典型上限 ~+3–6 stop),原生表达不了 10⁵ 的太阳——它更适合显示 HDR,作为 scene-linear EXR 的中间件而非 IBL 终目标。

**先例/参考**:https://www.iso.org/standard/86775.html ;GM-Diffusion https://github.com/Guanys-dar/GM-Diffusion ;GMODiff https://arxiv.org/abs/2512.16357

**风险**:需生成两张一致平面;极端比值处增益精度仍有限;组件更多。

> 其余方案(PU21 见 X2HDR、PQ、LogC3/4、Log-Gamma、μ-law、filmic 家族、PBR Neutral、LatentHDR、可学习 RQS 样条)的公式与取舍见文末**附录 A:编码曲线速查**。

### 双目标张力的核心结论(这是区域 1 最该记住的)

- **"往返保真"与"扩散可学性"相关但不同,耦合变量是编码 + 归一化把中灰和长尾放在哪里。**
- 中灰(0.18 线性)在 [0,1] 的落点:sRGB=0.461(VAE 的世界,目标)、LogC3=0.391✓、S-Log3~0.41✓、**Log3G10~0.33**、PU21~0.5、**PQ@4000nit~0.71✗**、线性=0.18✗。
- **相机 log 按构造同时解决两者**;PU21 只赢往返;PQ 最大头顶空间但 off-manifold。
- DiffHDR 消融佐证你们的冻结直觉:**微调 VAE 会过度平滑、抑制高频**,故全部负担压在编码 + 去噪器 LoRA。

### 归一化 —— 单个最重要的决定

- **用固定物理锚点:18% 灰 = 0.18 线性。绝不用逐图 max**(逐图 max 摧毁绝对辐射 = 使 IBL 强度失去意义 = 从根上违背 EXR 目的)。若必须逐图自适应,把 scale 写进 **EXR 元数据**以精确反变换。
- 峰值**刻意压在 V=1 以下**留 headroom(+6 stop over gray = 0.18·64≈11.5 线性,落在 V≈0.85–0.9)。

### 训练 LoRA 前必做的廉价决定性实验(半天)

> 取一批真 EXR,分别用各候选曲线做 **encode → VAE → decode → 反变换**,测**线性域 PSNR/ΔE + 仅高光 PSNR + 每 stop 码值预算**。这复现 X2HDR 的往返测试,**在你自己的 Qwen VAE 上**把 Log3G10 vs LogC3 vs asinh vs PU21 vs PQ 排序。这一个实验就决定了"冻结 vs 改造"对你的峰值目标是否够用。

### 技术顾问建议(区域 1)

1. **MVP 路径**:Log3G10(或 X2HDR 原样的 PU21,作为最快对齐已发表结果的基线)+ 冻结 VAE + 单 LoRA,配固定 18%-gray 锚点。先跑上面的往返实验选曲线。
2. **终极路径**:仍以相机 log/asinh 为编码,但叠加 **LumaFlux 式单调 RQS 样条 post-decode 头**杀掉 banding;对太阳量级需求切到**增益图双平面或 LatentHDR 曝光栈**(把极端范围放单独平面)。
3. **实施顺序**:往返实验 → Log3G10 LoRA(RAW→Turbo)→ 若 banding 不可接受则加 RQS 样条头 / asinh → 需要太阳量级再上双平面。

### 意外发现(区域 1)

- **RED Log3G10 从未被 HDR-diffusion 论文用过**,但按中灰落点 + 头顶空间 + 负值稳定三项分析它应优于当前 SOTA 用的 LogC3——一个低风险、可能直接提升的点。
- **asinh 拉伸**是天文界几十年的标准,近黑行为优于纯 log,却没人搬进 HDR 生成。
- **LumaFlux(2604.02787)** 的单调 RQS 样条 tone-field 解码头,是"在不动冻结 VAE 前提下按学习方式扩范围并防 banding"的现成范式,直接对症 Sumit 报告的 banding。

---

## 探索区域 2:在 Qwen Image VAE 上"冻结 vs 改造"的取舍与中间地带

> **先纠正一个前提(第一手读源码)**:Qwen VAE 解码器**没有**学习到的有界激活,只有 `_decode` 里一行 `torch.clamp(out,-1,1)` + 输入把 `[0,1]→[-1,1]`。去掉 clamp 一行搞定、不碰权重。**所以解码器不是瓶颈,编码侧才是**——冻结编码器从没学过 >1 高光,喂原始线性 >1 = 越界。**解 crux(区域 2 的 Q6)**:只去 clamp **不够**;必须先把范围压进编码器的分布内(log/PQ),反变换放在解码器之后。

### 发现的方案全景

1. **1A 固定可逆变换 + VAE 100% 冻结**(=区域 1 路线,只训 DiT LoRA)。
2. **1B 可学习逐通道 prenorm + 去解码器 clamp + 编码器仍冻结**。
3. **2 LoRA 化 VAE 微调**(对 VAE 卷积/归一化层低秩适配,而非全量)。
4. **3 潜通道 16→32 双头 + 零初始化 DiT 适配**(DA-VAE 式最小侵入)。
5. **GM 增益图旁支**(第二潜分支,骨干基本冻结)。
6. **Chatterjee 全改造**(编码器解冻 + 16→32 + 全 MMDiT LoRA)——保真上限,但放弃免费 Turbo 迁移。

### 方案详情

#### 方案 2.1A:固定可逆变换 + VAE 全冻结 + 只训 DiT LoRA
见区域 1 方案 1.4/1.6。**唯一干净保住官方 RAW→Turbo LoRA 工作流的方案**(改数据流形而非张量接口)。难度低。真实 >1,曲线限定峰值。

#### 方案 2.1B:可学习逐通道 prenorm + 去 clamp,编码器冻结

**一句话原理**:用一个小的**可学习输入变换**(逐通道仿射,或微型单调 MLP)把场景线性映进编码器最佳区,并**删掉解码器 clamp**,让输出端反变换能吐 >1。

**灵感来源**:Sumit 的可学习逐通道归一化——但**编码器保持冻结**(他解冻);FastVSR/非对称 VAE 解码器侧微调先例(2509.24142)。

**具体实现步骤**:
1. 插 `PreNorm`:`z_in = a⊙f(x_lin)+b`,f 为固定 log,a,b 逐通道可学习(或 2 层单调 MLP)。
2. 插 `PostDeNorm` = PreNorm 的解析逆,并**删/放宽 `torch.clamp(-1,1)`**。
3. 冻结编解码卷积,只训 prenorm/postdenorm(+ 可选解码器 LoRA),损失在 **log/感知空间**(线性域 L2 会被高光支配而崩)。
4. 再照 1A 训 DiT LoRA。

**与现有系统对接**:潜空间近原生 → **保迁移**(把 a,b 正则向固定-log 恒等靠)。难度低–中。可学习曲线能比固定曲线更好地把码值分配给高光 → 同峰值下 banding 更少。

**风险**:可学习曲线非单调 → 不可逆(用 softplus 参数化强制单调);只去 clamp 而不做前置压缩 = 无效(见 crux)。

#### 方案 2.2:LoRA 化 VAE 微调

**一句话原理**:不全量微调 VAE,只往卷积/归一化层注入低秩适配器,最小、可逆地扰动权重(与潜空间几何)。

**灵感来源**:**GM-Diffusion(ICCV 2025)** 就对 SD VAE 做 LoRA 来预测增益图;Conv-LoRA SR(2504.11271);变码率压缩 VAE-LoRA(2606.16107);FVAE-LoRA(2510.19640)。——**这条是真有人做过的。**

**具体实现步骤**:
1. 给 VAE `Conv2d/CausalConv3d`(可选 groupnorm 仿射)包 LoRA(`ΔW=BA`,rank 4–16,**B 零初始化**→ 第 0 天=恒等)。
2. 范围:**仅解码器 conv-LoRA**(最安全,配 2.1B)或**编解码都上**(想让编码器真正重分配高光容量时)。
3. 去解码器 clamp,EXR 上以 log/感知损失训练 + 一个 **KL/潜统计正则**(惩罚潜均值/方差偏离 base `latents_mean/std`——这是保 DiT 兼容的关键)。
4. 合并或留作适配器;再训 DiT LoRA。

**与现有系统对接**:**中等风险、可调**。零初始化 + 低秩 + 潜统计正则让潜分布贴近原生,则 RAW→Turbo DiT LoRA 仍迁移。但**任何编码器侧适配都会移动潜流形**;rank/范围越大,DiT(及记住旧流形的蒸馏 Turbo)看到的输入越偏 → 迁移退化。编码器 LoRA rank 保持极小或干脆冻结。**监控 `KL(latent‖base)` 以防悄悄弄坏 Turbo。**

**难度评估**:中。接线简单,难点在潜稳定正则与损失空间。

**预期效果**:**编码器 LoRA 参与时为真实**(编码器获少量容量保住高光结构)→ 同峰值 banding 少于方案 1。仅解码器 LoRA = 与方案 1 同上限,主要买重建保真。

#### 方案 2.3:潜通道 16→32 双头 + 最小侵入 DiT 适配

**一句话原理**:为线性/HDR 分量新增 16 个潜通道,用**零初始化的 patch-embed 与 final-layer 行**把它嫁接到**已训练好的 DiT** 上,使第 0 天行为与 16 通道逐位相同,新通道以残差方式学出来。

**灵感来源**:**DA-VAE(2603.22125,2026-03)**:为适配新潜通道,加一个**额外 patch embedder + 额外输出头、零初始化**,"第 0 步数学上与原模型相同",再配**余弦退火的渐进损失**(早期压低新通道损失、逐渐拉起)。Sumit 的"复制现有行 16→32"是同思想的粗版。

**具体实现步骤**:
1. VAE:训一个 32 通道 VAE,通道 0–15 复现**原**tonemap 潜码(正则去匹配冻结 base 的 16 通道输出——这是保迁移的关键),16–31 承载线性/HDR 残差(双头解码 → 对齐的 tonemap+linear)。
2. DiT patch-embed:原 16→hidden 投影**冻结/复制**;新增 16→hidden 投影**零初始化**;两者相加。
3. DiT final layer:加**零初始化输出头**产出 16 个新通道速度。
4. 渐进损失调度 + 高光加权训练。可选:冻结整个原 DiT,只训两个零初适配器 + 小 LoRA = "部分通道 LoRA"。
5. 先 RAW,再移植到 Turbo。

**与现有系统对接**:**这里迁移变难**。即使零初适配,训完后 DiT 的有效输入是 32 通道;**Turbo 是在 16 通道上蒸馏的、根本没有 HDR 通道**。故 RAW 训的 32 通道适配器**不能直接落到原版 Turbo**。要么把零初适配器也嫁接到 Turbo 并短暂再蒸馏(方案 4),要么让原 16 通道**冻结不变**使 SDR 行为至少能迁移、只有 HDR 头需 Turbo 侧训练。**部分破坏——SDR 路径迁移,HDR 路径不迁移。**

**难度评估**:**高**。需训新 VAE + DiT 适配器 + 重碰 Turbo。与 Chatterjee 同级。

**预期效果**:**真实、保真最好**(专用高光通道 → banding 最少,共享潜码对齐 tonemap+linear,即 Sumit 的 >500 峰值)。EXR 意义太阳量级:可达。

#### 方案 GM:增益图旁支(见区域 1 方案 1.10)
GM-Diffusion 达 >4000 nit(BT.2020)、骨干基本冻结、**无需 HDR 基座数据集**(在 MSCOCO 上训)。但显示参考(nits)非场景线性辐射 → 对 VFX EXR/Nuke 目标不够原生。若交付物是显示 HDR 则是低破坏路径。

### 中间地带的核心结论

- **去 clamp ≠ HDR**(Felldude 负对照证明:错目标的解码器微调 = 锐化);
- **冻结编码器确实丢 >1 信息**(对原始线性),但**对已压进其训练范围的信息不丢**;
- 故最干净的中间地带**不是解码器手术,而是"前置范围压缩(log/PQ)+ VAE 全冻结 + 单 DiT LoRA"**;
- Chatterjee 的编码器解冻 + 16→32 是保真天花板,但放弃免费 Turbo 迁移,**只在 log 空间 banding 在你的目标峰值不可接受时才往上爬**。

### 决策矩阵

| 方案 | 编码器 | 潜维 | DiT 改动 | 破坏 RAW→Turbo? | 真 HDR? | 峰值 | 难度 |
|---|---|---|---|---|---|---|---|
| 1A log 编码全冻 | 冻结 | 16 | 仅 DiT LoRA | **否** | 是 | 曲线限(PQ 10k nit / LogC 几百) | 低 |
| 1B 可学 prenorm+去clamp | 冻结 | 16 | DiT LoRA | 基本否 | 是 | 曲线限,高光分配更优 | 低–中 |
| 2 VAE conv-LoRA | 可 LoRA | 16 | DiT LoRA | 中(潜统计正则可调) | 是 | 曲线限,banding 更少 | 中 |
| 3 16→32 双头 | 重训 | 32 | 零初适配+LoRA | **是(HDR 路径)** | 是,保真最好 | >500 | 高 |
| GM 增益图旁支 | LoRA | +分支 | 第二潜分支 | SDR 否/HDR 新 | 显示 HDR(nits)>4000 | 中 |
| Chatterjee 全改 | 解冻 | 32 | 全 MMDiT LoRA | 是 | 是 >500 | 高 |

### 技术顾问建议(区域 2)

1. **MVP**:方案 **1A**——唯一干净保住官方 RAW→Turbo 的选项。跑区域 1 的往返实验定曲线与上限。
2. **终极**:若 banding 不可接受,**原地升级 1B → 2**(加可学 prenorm,再上编解码 conv-LoRA + 潜统计正则),**再**才付出方案 3 / 全 Chatterjee(唯一给共享潜码对齐 tonemap+linear,但放弃免费 Turbo 迁移、需 Turbo 侧再蒸馏)。
3. **顺序**:1A →(banding 驱动)1B → 2 →(需双输出/最高保真)3。

### 意外发现(区域 2)

- **"[0,1] 天花板"不是学出来的 tanh,而是一行 `clamp` + 输入归一化**——去掉它是一行代码。这把问题从"解码器手术"重新定位到"编码侧信息注入",极大简化了工程判断。
- **conv-LoRA-on-VAE 是真实、近期、可行的**(GM-Diffusion / 2504.11271 / 2606.16107 / 2510.19640),给了"改造 VAE 但最小破坏迁移"的中间档。
- **DA-VAE 的零初始化通道嫁接 + 渐进损失**是把已训练 DiT 适配到新潜通道的现成配方,让方案 3 不必从头重训 DiT。

---

## 探索区域 3:HDR 训练数据的获取与合成

### 发现的方案全景

1. **S2R-HDR(你们指定的候选)** —— 场景线性干净,但**透视非全景、未定标太阳、无 caption**;作"卫生/监督"来源而非 IBL/太阳骨干。
2. **Poly Haven** —— ⭐ **最佳公共来源**:978 张真实包围曝光**等距柱状全景** HDRI,最高 24K,**CC0**,户外保留真太阳。
3. **Laval 家族**(Outdoor 205 / Sky 40k+ / Indoor 2100 / Photometric / PanDORA) —— 真太阳辐射最丰富,但**仅研究授权**。
4. **Fairchild HDR Photographic Survey** —— 106 张 32-bit 光度线性 EXR,透视,**公有领域**,最干净的小补充。
5. **程序化生成**:Blender+Nishita(全景+物理太阳)、UE5+XRFeitoria、Infinigen、Hypersim/OpenRooms(室内无太阳)。
6. **从真 HDR 合成 LDR 退化配对**(Poisson-Gaussian 噪声 + 真 CRF)。
7. **逆色调映射预生成伪 HDR** —— 对太阳/光源=**毒药**,仅对 <1–2 stop 漫反射高光可接受。
8. **跨领域**:HDR 视频(PQ 显示参考,慎用)、遥感/天文/显微(辐射但严重 OOD,**跳过**)。

### 方案详情

#### 方案 3.1:S2R-HDR 的正确用法(卫生来源,非骨干)

**核实规格**:arXiv 2504.07667,ICLR 2026,OpenImagingLab。24000 帧 = 1000 序列 ×24,1920×1080 **透视**,UE5 Lumen(XRFeitoria)渲染、**tonemap 与 gamma 均关闭**的场景线性 EXR,附 flow/depth/normals/albedo/camera_params,CC-BY-4.0,**无 caption**。目标是**多曝光 HDR 融合/去重影**,不是生成也不是 IBL。

**评估**:场景线性卫生**极佳**;太阳量级**弱/未定标**(帧内 DR ≈3.86 log10 ≈12.8 stop,"含阳光"但无 nits/lux 定标,很可能远不及真太阳);**非全景**(不能直接做 IBL);有效场景 ~1000(24 帧近重复)。S2R-HDR-2 = 同发布后半,**非 v2**。

**具体用法**:
1. **先抽样检查真实 EXR 的最大值**,确认太阳是否被裁;
2. 用作**场景线性监督/统计对齐**来源(它的辐射线性性干净),不作 IBL/太阳骨干;
3. **加 caption**:曝光归一 + tonemap 成良曝 LDR → VLM(Qwen-VL)描述,**融合已有的 UE5 渲染元数据**(场景 id、运动类型、昼/暮/夜、室内外、物体类)得可靠结构化 caption。

**风险**:合成→真实域隙(该数据集自带 S2R-Adapter 就是为跨这个隙,你会遇到类似问题);非全景;太阳未定标。

#### 方案 3.2:Poly Haven + Fairchild + 程序化(推荐的骨干数据组合)

**一句话原理**:用"真实全景真太阳(Poly Haven)+ 光度线性透视(Fairchild)+ 可控极值的程序化全景(Blender/Nishita)"三者拼出**既有真辐射、又覆盖太阳量级、又 IBL-ready** 的训练分布。

**具体实现步骤**:
1. **Poly Haven**:抓 978 张 CC0 等距柱状 HDRI(API `https://api.polyhaven.com/assets?type=hdris`),最高 24K,户外真太阳。既做全景 IBL 子集,也**投影成随机透视裁剪**匹配自然图取景(LEDiff/LuxDiT 就这么做)。
2. **Fairchild**:106 张公有领域光度线性 EXR 作干净小补充(附绝对亮度乘子)。
3. **Blender+Nishita 程序化全景**(保证 10⁴–10⁹ 太阳极值):见方案 3.6。
4. **仅 Laval**:若能拿到研究授权再加(太阳辐射最丰富)。

**授权**:Poly Haven CC0(可商用)、Fairchild 公有领域(非商研究)、Laval 研究 EULA、S2R-HDR CC-BY-4.0。

#### 方案 3.6:程序化 scene-linear EXR(唯一能保证太阳极值的来源)

**Blender + Nishita(全景+物理太阳,首选)**:
1. **场景线性**:输出 **Multilayer OpenEXR Float(Full) 32-bit**——EXR/HDR/DPX **忽略 View Transform**,故 Standard/Filmic/AgX 产出同样的线性像素。Color Management **Exposure=0, Gamma=1**(注意:"Save as Render" 开启时这俩滑块可能烘进 EXR——**务必用已知辐射值验证**)。
2. **物理太阳**:World → Sky Texture → **Nishita**,开 Sun Disc。**已知坑:Nishita 的盘面比等效 Sun lamp 暗约 10×(dev T79249)**;要保证 10⁴–10⁹ 极值,**用显式 Sun lamp**(Strength 用 W/m²,角径 0.526°)做盘面 + Nishita 做天空散射——最可控、物理上可辩护。
3. **全景**:Camera → Panoramic → **Equirectangular**,2:1(≥4096×2048),存 32-bit EXR,直接作 HDRI。
4. **批量/标注**:Python/BlenderProc/Infinigen 驱动,caption 从程序化参数(太阳高度、天气、资产表)或 VLM 生成。

**UE5 + XRFeitoria(可扩展透视,S2R-HDR 路线)**:MRQ → .exr 勾 **Disable Tone Curve**(+ `ShowFlag.Tonemapper 0`、`EyeAdaptation 0`、关 bloom);Directional Light 用 lux(晴天正午 ~100000–120000 lux)+ Sky Atmosphere。仅透视 → 360 IBL 仍用 Blender。

#### 方案 3.6b:LDR 退化配对管线(把真 HDR 造成配对)

**关键洞见**:大多数 HDR 论文遗漏的一步是**在线性域、CRF 之前**加 **Poisson-Gaussian(散粒+读出)噪声**——这是让加曝阴影变噪、最提升真实感的一步。

**推荐有序管线**(对线性辐射 H):
1. (可选)逆白平衡抖动(逐通道增益 U[0.75,1.25]);
2. **曝光/增益**:v~U[0.05,0.20],`s = 1/percentile_{1−v}(luminance)`,H←s·H,±2EV 抖动;
3. **线性域 Poisson-Gaussian 噪声**:`y~N(x, λ_read+λ_shot·x)`,ISO 增益 log g_a~U[log1,log16](Foi / Brooks "Unprocessing" CVPR19);
4. **应用真 CRF**:从 **DoRF via EMoR(11-PCA 基)**采样(金标准),轻量退路=HDRCNN sigmoid 或 gamma∈[1.8,2.4],逐样本随机;
5. **裁剪/饱和**:`I=clip(f(H),0,1)`,I≥0.95 处记饱和 mask → 喂损失(高光当 inpainting);
6. **8-bit 量化**,可选 JPEG q85–100。
HDR ground truth 保线性单位,训 log/µ-law 损失 + clipped-mask + 高光项。

### 数据量与 VAE 阻断器(核心结论)

- **内容/风格 LoRA 便宜**(~10–50 图,T-LoRA 2507.05964 甚至单图)。**但把输出分布扩到 HDR 辐射是另一回事,且触及 VAE**:
  - LEDiff:**36000** HDR 图,**全量微调**(VAE 解码器 200k 步 + 去噪器 400k);
  - Sumit:**~9000 EXR**,全量微调 VAE(16→32)+ rank-64 MMDiT LoRA;
  - X2HDR:小数据,**LoRA-only VAE 冻结**——**正因为**输入预编码到 PU21/PQ;
  - LatentHDR:Poly Haven + SI-HDR 小数据,LoRA。
- **VAE 范围问题是真的**(两个独立 2026 来源证实):线性 >1 喂进 stock LDR VAE = "不确定的垃圾"(Sumit)/"严重退化"(X2HDR)。**"去噪器风格 LoRA + VAE 不动 + 直接喂线性 EXR"必然失败。**
- **两条逃逸,先选一条**:
  - **Path A**:VAE 前编码进有界感知容器(PU21/LogC4/log),输出反变换回线性。**LoRA 尺寸、VAE 冻结、数据最少(数十–数百)。** PQ 是显示参考 → 仅作可逆容器,始终往返回场景线性。
  - **Path B**:微调 VAE(解码器或倍增通道)原生扩范围。**~9k–36k 图。**

### 场景线性 vs 显示参考(PQ)不能混用

- 场景线性(ACES2065-1/ACEScg,EXR):值∝辐射、**无上界**、太阳几百–几千;显示参考(HDR10/PQ ST2084/HLG):绝对 nits、**PQ 硬顶 10000 nit**、高光不可逆裁剪。
- 混用毒化:"值=2.0" 同时意味"2× 辐射"和"~700 nit 码"→ 矛盾目标、浑浊高光、破坏 IBL 依赖的线性比。
- **元数据 schema**(以 OCIO/ACES role 为准):每图存 `transfer_function`(scene-linear/PQ/HLG/sRGB/LogC4/PU21)、`encoding`(场景 vs 显示参考)、`primaries/gamut`、`luminance_scale`(绝对 nits 乘子或 EV 归一)、`container_transform`(若把场景线性再编码进 PU21/PQ/LogC4 作容器,**单独记录**,别与真显示参考混淆)。**只 ingest 场景线性分区**;PQ 素材先 inverse-EOTF + 可辩护逆 tonemap 转场景线性并**标记为 reconstructed**。

### 逆色调映射/伪 HDR 风险

- **你最需要的极值(太阳、镜面)恰是 ITM 无法恢复的**:裁到 255 辐射就没了,任何"重建"都是学出来的猜。Deep HDR Hallucination(PMC8230591):"太阳的形状"不会被重建;AIM 2025 ITM(2508.13479):"在饱和高光下不适定"。
- **可接受**:恢复轻裁的漫反射高光/窗边/轻微天空梯度(<1–2 stop,邻域约束)——作增强/软目标。**毒药**:把捏造的太阳/光源强度当 ground truth 教给生成模型 → 模型学到 ITM 网的偏置当真辐射,污染任何辐射积分下游用途(IBL/relight)。若必须,**从监督里 mask/剔除裁剪光源**或换定标参数化光,绝不让猜出来的太阳成为真值。

### 捕获/标注(scene-linear → T2I)

- VLM 是 LDR 原生 → **tonemap EXR→LDR(AgX/filmic/Reinhard)→ VLM 描述(Qwen2.5/3-VL 领先,CogVLM/InternVL/LLaVA)**,再增强。
- **HDR 专属增强(注入 LDR 代理丢掉的信息)**:持久触发词 `"scene-linear HDR"`/`"HDRI"`/`"high dynamic range"`(LoRA 激活短语);VLM 从 tonemap 看不到的**辐射元数据**:EV/曝光 stop、太阳存在 + 高度/方位、峰值亮度/动态范围比、室内外、主光描述。可选:对多曝光分别 caption。

### 技术顾问建议(区域 3)

1. **先定 VAE 策略**(Path A vs B),它决定数据量级。第一步计划是小 LoRA → 走 **Path A**,数据仅需数十–数百。
2. **骨干数据**:Poly Haven(CC0/全景/真太阳)+ Fairchild(公有领域/场景线性透视)+ Blender/Nishita 程序化全景(显式 Sun lamp 保 10⁴–10⁹)。
3. **S2R-HDR 作场景线性卫生源**(先验峰值),用 UE5 元数据 + tonemap→VLM 加 caption。
4. **只训场景线性**(严格 transfer-function 标签),PQ 除非转换+标记否则排除。
5. 需配对则用上面的 Poisson-Gaussian + DoRF/EMoR 管线;把 360 投影成透视裁剪并保留全景子集给 IBL 微调。

### 意外发现(区域 3)

- **S2R-HDR 虽是你们指定的数据集,但用途要重定位**:它天生为 HDR 融合(多曝光去重影)造,不是为生成/IBL,且透视、未定标太阳——**对本项目它是"场景线性卫生/统计对齐"来源,不是主训练骨干**。真骨干应是 Poly Haven + 程序化。
- **Blender Nishita 太阳盘面暗 10× 是个隐坑**:若直接用 Nishita 出 IBL,太阳能量会系统性偏低 → 用显式 Sun lamp 才能定标正确。
- **Poisson-Gaussian 线性域噪声**是多数 HDR 配对管线遗漏、却最影响真实感的一步。

---

## 探索区域 4:损失函数与监督空间

### 承重洞见(先说结论)

**对"冻结 VAE 上的扩散/流匹配 LoRA",本质上只有一个必要损失 = 潜空间的去噪/流匹配目标;编码曲线本身就是你的感知损失。** Hanji/Mantiuk(arXiv 2312.03640,RAW&HDR 训练权威研究)实测:PU21/PQ/µ-law 编码的 L1 比线性 L1 好 **2–9 dB**,且**再叠加第二个损失项(L1+PQ-L1)毫无额外收益**——编码**吸收**了感知损失。X2HDR/LumiVid/LatentHDR 训练时**只用标准流匹配损失、无任何额外感知/HDR 损失**。

### 发现的方案全景

1. **PU21/PQ/LogC 编码 + 普通 L1/L2**(即"选对编码空间")——主力,占 90% 价值。
2. **潜空间高光加权(masked)损失** —— Sumit 唯一实际加的额外项。
3. **ColorVideoVDP(`cvvdp.loss`)** —— VDP 家族里**唯一真能训练**的、HDR 原生 pytorch 损失。
4. **FLIP / HDR-FLIP** —— 另一个 loss-ready 感知损失(NVlabs 现成模块)。
5. **分布/log-直方图匹配高光** —— 对"几个太阳像素定一切"的原理性解法。
6. **分频段/分亮度区间加权、tanh/Reinhard 压缩、梯度域损失** —— 防高光支配的一组。
7. **µ-law PSNR** —— HDR 挑战赛事实标准指标(可作损失亦可作评测)。
8. **HDR-VDP-2/3、PU-PSNR/SSIM、PU-LPIPS/DISTS、TMQI、无参考 HDR IQA** —— **评测用**,不适合当损失(不可微或易被 hack)。

### 方案详情(节选)

#### 方案 4.1:编码空间即感知损失(主力)
**原理**:x0 = VAE.encode(Encode_HDR(radiance)),Encode_HDR∈{PU21,PQ,LogC3/4},损失 = 该潜码上的标准流匹配。**PU21 用 Pmax=595 归一而非 255(2405.00670)**。gfxdisp/pu21 是 MATLAB,~10 行可移植到 pytorch(输入 clamp >0.005 防 -inf)。难度低,风险:锚定 nits 必须编码与导出一致否则 EXR scale 漂移。

#### 方案 4.2:潜空间高光加权损失(唯一推荐的额外项)
**原理**:`w = 1 + λ·smoothstep(Lhi)` 逐元素乘到流匹配损失,mask 从 log 编码后的目标潜码导出,**无需 decode**。这是最接近你们场景的真实构建(Sumit)唯一加的项。风险:过度加权会**反向**重新引入太阳像素支配,调 λ。

#### 方案 4.3:ColorVideoVDP 作损失(唯一可训的 VDP)
**原理**:VDP 谱系的彩色+视频可见差异预测器,原生 pytorch,`gfxdisp/ColorVideoVDP` 暴露 `cvvdp.loss`(pip cvvdp)。HDR 原生、需告知峰值 nits 与色彩空间。**作者警告:"可能破坏优化景观凸性",建议配 L1、只在 L1 收敛后的后期低权重用。** 对 12.9B LoRA 是强告诫 → 极小权重、后期、decode 辅助项。

#### 方案 4.4:FLIP / HDR-FLIP 作损失
NVlabs/flip 提供 `LDRFLIPLoss`/`HDRFLIPLoss` 现成可微模块(pip flip-evaluator)。HDR-FLIP 吃线性 + tone_mapper(aces/hable/reinhard)+ start/stop_exposure。**坑:非对称,预测放第一参、参考放第二参否则梯度错。** decode 辅助项,每步一次曝光扫描有成本。

#### 方案 4.5:分布/log-直方图匹配高光
**原理**:不去逐像素回归几个太阳像素(不适定),而是匹配高光区 log 亮度的**分布**(可微 KDE 软分箱 / sliced-Wasserstein / MMD)。这是"几个太阳像素决定一切"的**原理性答案**:约束"有多少超 1.0 能量及其形状"而非单个像素。A Cycle Ride to HDR(2410.15068)。decode 辅助项。风险:不定位 → 高光可能可信但错位,配弱空间项。

### 真实损失架构(核心答案)

1. **主损失** = 编码潜空间的流匹配/v/eps(X2HDR/LumiVid/LatentHDR **别的什么都不加**);
2. 感知/HDR "损失"**通过编码而非损失项**进入;
3. 可选辅助①:潜空间高光加权(Sumit);
4. 可选辅助②:decode 精修**极省着用**(低权重、后期、每 N 步):cvvdp.loss / HDR-FLIP / PU-LPIPS / log-直方图 **四选一**,当抛光非骨干。
- **Krea2 推荐配方**:PU21 或 LogC 编码(选对 Qwen 潜统计 KL 最小的)→ rectified-flow + logit-normal t 采样(Qwen-Image 默认,**保留**)+ 适度高光加权 → 保 min-SNR 式加权 → 仅当评测见高光伪影才后期加一个 decode 辅助。**评测(非训练)用 PU-PSNR、PU-SSIM、PSNR-µ、HDR-VDP-3/cvvdp、PU-VTAMIQ。**

### 沉睡陷阱(重要、反直觉)

> **扩散噪声调度是为感知均匀的 8-bit 图标定的;在 log 空间里,均匀高斯潜噪声在 decode 之后变成偏向高光的非均匀噪声**(Chatterjee 指出)。编码修好了损失,但你必须**核查 decode 后噪声是否感知均匀**——支配问题可能从"损失"悄悄搬到"噪声调度/解码非线性"。考虑轻微 timestep 分布位移(Improved Noise Schedule 2407.03297:移动 timestep 分布优于重加权),而非新损失。

### 对抗损失 vs 扩散损失(长尾下)

- 若先编码,潜分布贴近 SDR → 标准流损失**行为正常**,无需特殊 HDR 项。
- GAN on HDR:在高方差高光尾**模式崩溃**(判别器只需骗过"真实感",乐得丢掉罕见亮模式=你要的长尾)。**扩散 LoRA 不需要对抗项,徒增不稳定**;要高光可信度,用分布匹配(4.5)而非对抗(直击尾部而非让判别器忽略它)。

### 技术顾问建议(区域 4)

1. **MVP**:选对编码(PU21/LogC)+ 标准流匹配 + 适度潜空间高光加权。**别加感知损失**(编码已吸收)。
2. **终极**:同上 + 后期极低权重的 **HDR-FLIP 或 cvvdp** decode 精修(仅当高光伪影出现)+ log-直方图分布项约束高光能量。
3. **顺序**:先只用编码空间损失跑通 → 评测(PU-PSNR/PSNR-µ/cvvdp)→ 有针对性地按需加辅助项。

### 意外发现(区域 4)

- **"损失几乎不用设计,编码即损失"** 是本区域最反外行直觉、也最省事的结论:任务书里"在 PU21/PQ/log 空间算 L1/L2"其实就是全部主力,叠更多项无益。
- **decode 后噪声非均匀**这个沉睡陷阱,可能是"编码对了但高光仍不稳"的真凶,值得作专门 sanity check。
- VDP 家族里**只有 ColorVideoVDP 能真的当损失**,其余(HDR-VDP-2/3、PU-LPIPS、TMQI、无参考 IQA)全是评测或 metric-hacking 蜜罐。

---

## 探索区域 5:ComfyUI 端"采样 → 反变换 → EXR"工程落地

### 两个承重发现(第一手读源码)

1. **标准/Qwen VAE 的 decode 默认就 clamp 到 [0,1]**。`comfy/sd.py` 里默认 `process_output = lambda image: image.add_(1.0).div_(2.0).clamp_(0.0,1.0)`,作用于 AutoencoderKL 家族(Qwen 用的就是)。`vae.decode()` 的 return 行不 clamp,但 `process_output` 在 decode 内部执行并硬裁 [0,1]。**这是 HDR 死掉的第一处,发生在任何保存节点之前。** 只有 StageA/audio/pixel/TAEHV 等特殊 VAE 设成恒等。
2. 两个主流 32-bit EXR 保存节点都能正确写、但都可能因色彩分支悄悄毁 HDR:HQ-Image-Save(`cv2.imwrite`,OpenCV EXR 默认 FLOAT/32-bit)的 `sRGB`/`Reinhard` 分支 `np.clip(0,1)`,只有 `linear` 分支直通;内建 `SaveImageAdvanced`(2026-05-25 合并,PyAV/ffmpeg,32-bit float,`linear`=原样写,`HDR`=逆 HLG)。

### 发现的方案全景 / 节点表

| 节点 | 库 | 位深 | 裁 >1.0? | 色彩变换 | HDR 结论 |
|---|---|---|---|---|---|
| 内建 SaveImage | PIL | 8-bit | **是** | 假设 sRGB | 毁 HDR,绝不用 |
| 内建 SaveImageAdvanced(CORE-32) | ffmpeg | EXR 32-bit float | `linear`/`HDR` 不裁 | linear/HDR/sRGB | 可用,`linear`=直通;压缩/属性控制有限 |
| **ComfyUI-HQ-Image-Save**(spacepxl) | cv2 | 32-bit float | `linear` 不裁;sRGB/Reinhard 裁 | linear 直通 | **用 `tonemap="linear"`**;SaveLatentEXR 写 4ch 潜码 RGBA 32bpc 不裁 |
| ComfyUI-EXR-API(nofunstudio) | OpenImageIO+OCIO | 32-bit | linear 支持 >1 | OCIO 转换 | OIIO 派、多层/AOV,save 部分曾 WIP |
| ComfyUI-CoCoTools_IO | OIIO | 32-bit | linear 支持 >1 | Nuke-Shuffle 式 | EXR **加载**强(层/AOV/cryptomatte),saver 曾 TODO |
| **ComfyUI_Gear**(oumad,2026-05-24) | 纯 python | **float16** | 不裁 | **LogC3/LogC4 decode→SaveEXR**、ACEScct | **最直接相关**(就是"反编码到线性再写 EXR"),但 float16 有溢出风险 |
| Marigold SaveImageOpenEXR | OpenEXR | 32-bit float | 无 | 主要 depth/normal | 可用于浮点 |
| Luminance-Stack-Processor HDRExportNode | — | 32-bit float | 无归一化 | 线性辐射 | 专为 HDR EXR |

### 方案详情

#### 方案 5.1:VAEDecodeHDR 自定义节点(去掉 process_output 的 clamp)
```python
class VAEDecodeHDR:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}
    RETURN_TYPES = ("IMAGE",)
    def decode(self, vae, samples):
        orig = vae.process_output
        vae.process_output = lambda image: image.add_(1.0).div_(2.0)  # 保仿射,去 clamp_
        try:
            img = vae.decode(samples["samples"])   # NHWC float32, 未裁
        finally:
            vae.process_output = orig
        return (img,)
```
**>1.0 在哪出现决定了什么必须活过 clamp**:若在编码空间(PU21/PQ/log)生成,decode 出来本就 ~[0,1],>1.0 只在反变换节点之后出现(仍建议去 clamp 以防顶部重建 overshoot 被硬裁);若直接生成线性(罕见),clamp 是灾难性的、必须去。

#### 方案 5.2:SaveEXR(OpenImageIO)自定义节点
```python
import OpenImageIO as oiio, numpy as np
def save(self, images, filepath, pixel_type, compression):
    arr = images.detach().cpu().float().numpy()          # (B,H,W,C) 不裁
    td = oiio.FLOAT if pixel_type=="float" else oiio.HALF
    for b in range(arr.shape[0]):
        px = np.ascontiguousarray(arr[b]); H,W,C = px.shape
        spec = oiio.ImageSpec(W,H,C,td)
        spec.channelnames = ("R","G","B","A")[:C]
        spec.attribute("compression", compression)        # zip=无损默认
        spec.set_colorspace("scene_linear")               # 仅元数据,不施曲线
        spec.attribute("chromaticities", oiio.TypeDesc("float[8]"),
                       (0.64,0.33,0.30,0.60,0.15,0.06,0.3127,0.3290))  # Rec.709
        out = oiio.ImageOutput.create(path); out.open(path, spec)
        out.write_image(px); out.close()                  # 原样写浮点,无 gamma 无 clamp
```
ACEScg 则 chromaticities 用 AP1 `(0.713,0.293,0.165,0.830,0.128,0.044,0.32168,0.33767)`。`set_colorspace` 在 OIIO 原始写路径**只是元数据、不施曲线**——正是你要的:标为线性、不变换。

#### 方案 5.3:反编码到线性节点(decode 后、SaveEXR 前)
- **PU21 逆**:`V=V_norm*Vmax; L=2^((2a·Lmin−b+sqrt(b²+4aV))/(2a))`,a=0.001908,b=0.0078,Lmin=log2(0.005),Vmax=编码器在 4000cd/m² 处的值。完整 gfxdisp PU21 `banding_glare` 参数见附录。
- **PQ→linear**:`Np=N^(1/m2); L=((Np−c1)+/(c2−c3·Np))^(1/m1)·10000`,m1=0.1593017578125,m2=78.84375,c1=0.8359375,c2=18.8515625,c3=18.6875。
- **LogC3 逆**:`L=(10^((V−d)/c)−b)/a`(常数见附录);LogC3 天花板 ~55 线性(会裁太阳),LogC4 ~470,故要太阳量级用 LogC4/PQ/PU21。

#### 方案 5.4:图内 EXR 往返自检(区分"节点吃了范围" vs "模型没生成范围")
```bash
oiiotool --stats out_0000.exr      # 每通道 Min/Max/Avg + NONFINITE_COUNT
oiiotool --info -v out_0000.exr    # 确认 float(非 half)+ compression: zip
```
```python
# 图内断言节点
buf = oiio.ImageBuf(path); st = oiio.ImageBufAlgo.computePixelStats(buf)
file_max = max(st.max)
if file_max < tensor_max*0.999: raise RuntimeError("保存路径 clamp/量化了 — 文件丢了动态范围")
if tensor_max <= 1.0001: print("警告:源张量从未超过 1.0 — 模型/反变换没生成 HDR(不是保存节点的锅)")
```
**这干净地二分故障**:file_max < tensor_max ⇒ 节点故障;tensor_max ≤ 1 ⇒ 生成/反变换故障。

### half vs full / 压缩 / 色彩管理

- **half(上限 65504,太阳可能溢出成 +Inf,~10bit) vs full(~3.4e38,~24bit)**:AI 管线**默认 32-bit float 主档**(模型反变换后可能 >65504)。溢出检测:`m=nan_to_num(arr).max(); m>65504`。half 只作下游尺寸优化交付且需验证 max 合适 + 有意 rolloff(**绝不硬 clamp**)。
- **压缩**:主档 = **ZIP 或 PIZ 无损 32-bit**。**DWAA/DWAB 是 DCT 有损,恰好劣化高频高幅高光(太阳边缘/镜面)产生 ringing → HDRI/IBL 主档避免**;PXR24 有损 24-bit 浮点不作主档。
- **VFX 色彩管理约定**:工作空间场景线性、**无 ICC**;显式设 `chromaticities`(缺省则应用假设 Rec.709);`oiio:ColorSpace="Linear"`。Blender:HDRI 图像节点用 linear/scene 空间(**Non-Color 对照明是错的**);Nuke:Read colorspace=linear;Unreal:sRGB 标志**关**、压缩 HDR/float;Unity:HDRI LatLong 类型 HDR 绕过 sRGB。**金律:渲染器照明用的文件 = 32-bit float、ZIP/PIZ 无损、场景线性、正确 primaries、零 gamma、零 clamp。**

### 技术顾问建议(区域 5)

1. **优先动作**:(a) 替换/patch VAEDecode 去掉 `process_output` 的 `.clamp_(0,1)`——这是沉默的第一杀手;(b) decode 后插反变换节点(按训练空间 PU21/PQ/LogC);(c) 用 OIIO 32-bit float ZIP scene-linear 保存,或 HQ-Image-Save `tonemap="linear"`(**绝不 sRGB/Reinhard**);(d) 加 `oiiotool --stats`/`computePixelStats` 往返断言二分故障。
2. **现成起点**:ComfyUI_Gear(oumad,2026-05)已实现 LogC3/LogC4 decode→EXR,是最接近你们需求的现成节点(但改成 32-bit)。

### 意外发现(区域 5)

- **VAE decode 自带 clamp 是任务书没意识到的"第一杀手"**——它在你的任何保存节点之前就把 HDR 裁了。任务书只担心保存节点,漏了这处。
- **往返自检节点能把"节点吃范围"和"模型没生成范围"两类问题干净分开**,这对调试极有价值(否则你会分不清是工程 bug 还是模型没学会)。
- **ComfyUI_Gear(2026-05)** 几乎就是你们要的"反编码→写 EXR"链路的现成实现,值得直接借。

---

## 探索区域 6:我们没想到的方向 + 整条路线是否走偏

### 直接的诚实判断(点 6,最高价值)

**计划的第一步会产出令人信服的 ~10–14 stop、显示参考的"HDR 电视级"输出;它不会可靠产出辐射正确、覆盖太阳的场景线性 EXR。这不是调参问题,而是烙进了:(a) 冻结解码器在压缩编码下的精度预算,(b) 裁剪区绝对尺度的信息论不适定性。**

- **信息论天花板没有硬 [0,1] 墙,但有两道软天花板**:
  - **天花板 1(表示)**:Qwen VAE 8× 压缩 + 为显示流形调的有效精度。把 ~20 stop 塞进 [0,1] → 曲线顶部极陡 → PU21/log 空间里微小 decode 误差在**线性域近太阳处放大成乘性巨误差**。decode 误差在感知空间有界、在线性域顶部几 stop **爆炸**。故 X2HDR 停在 ~14 stop/4000nit,冻结 VAE 包围融合只到 ~4.5 stop。**冻结 VAE 就是顶端的约束。**
  - **天花板 2(信息/不适定)**:任何被裁的区域真辐射不可恢复,是幻觉(Deep HDR Hallucination 2106.09486,AIM2025)。对电视可接受;对 IBL,"捏造的太阳强度"=错误场景辐照度=VFX/IBL 恰恰不能容忍的东西。**没有编码曲线能修**;取决于数据 + LoRA 训在什么上。训在 iTM/4000nit 显示 HDR → 保证伪 HDR;训在真场景线性 EXR(Poly Haven/路径追踪)→ 可信辐射但裁剪区绝对太阳尺度仍是猜。

### 最近 6 个月(2026-01~07)新工作(已验证 arXiv ID)

| 论文 | arXiv | 一句话 | 与你们的关系 |
|---|---|---|---|
| LatentHDR | 2605.11115 | 冻结 VAE + 独立条件潜到潜曝光变换,场景线性文/图→全景 HDR,SOTA 范围、~10× 更省算力 | 最接近的**竞品**,主张不需包围采样 |
| HDR 视频 via 潜对齐/LogC3(≈LumiVid) | 2604.11788 | LogC3→冻结 VAE→float16 EXR,PolyHaven+Tears of Steel 训练 | 自认"质量取决于输入表示能否忠实编码全辐射范围",反复用 "hallucinate" |
| **线性图像生成 via 合成曝光包围** | **2604.21008**(S-Lab NTU+Adobe) | K=4 包围(EV−4/−2/0/+2)过**冻结 VAE** 融合场景线性 | **关键反证**:明说冻结 VAE"难以同时保住极端高光和阴影",融合仅达 ~23× 比(~4.5 stop)。这是整条子领域改用包围的原因 |
| ExpoCM | 2605.02464 | 一步(一致性模型)HDR 重建 | 两阶段 SDR→HDR 精修器 |
| GMODiff | 2512.16357 | 扩散先验一步精修增益图 → 场景线性 | 增益图路线,直接相关 |
| RawGen | 2604.00093(Samsung/Yonsei) | FLUX.1-Kontext + 微调 VAE 解码器 sRGB→场景参考线性 XYZ | 跨域类比;**警示:仍把 XYZ 裁到 [0,1]**,靠 DNG 元数据给绝对尺度 |
| Qwen-Image-VAE-2.0 | 2605.13565 | 新 VAE,压缩翻倍、步数 40→4 | 若换基座 VAE 需重估 |
| HDR 环境图估计 via 潜扩散 | 2507.21261 | ERP-aware LDM HDRI | LuxDiT 的 HDRI 估计同胞 |
| AIM 2025 逆色调映射挑战报告 | 2508.13479 | 基准整个 iTM 领域 | iTM 天花板最佳参考 |

已验证既知:X2HDR 2602.04814、LEDiff 2412.14456、LuxDiT 2509.03680(NeurIPS25)、DiffusionLight 2312.09168/2507.01305。

### 原生 HDR 基座:开放权重里不存在

- 每个"HDRI 生成器"都是 retrofit 或 LDR 生成 + 逆 tonemap。**LuxDiT(NVIDIA)** 是最强原生 HDR 环境系统,但它**估计**光照非文生 HDR;且它把环境图编成**两张 tonemap 表示**、冻结 VAE、沿通道拼、DiT 融合——**NVIDIA 也不信任单次冻结 VAE**,用了双/冗余表示(呼应 Sumit 16→32)。**信号:冻结单潜路径是弱版本。**
- Text2Light(2022):4K HDR 全景但 stage2 = MLP 逆 TMO + stage3 参数化 FDR boost = 伪 HDR。DiffusionLight:冻结 SDXL+LoRA+**曝光包围**铬球(领域标准 = 包围再合并,因基座无法直接吐 >>1)。Blockade Skybox/LDM3D/EdgeRelight360:仍受 SD-VAE 有界。
- **无 EXR 原生/RGBE/LogLuv 潜自编码器基座**。社区 "Radiance VAE" ComfyUI 节点是未证实的营销。
- **结论:"换原生 HDR 基座"在今天开放权重里不是可选项。真正的分叉是"冻结 VAE vs 微调/扩容 VAE",每个严肃的原生范围系统(LuxDiT、Sumit)都选了后者。**

### Qwen/SD VAE 承载非 [0,1] 信号的证据(支持前提,带关键 caveat)

- **Marigold(CVPR2024,2312.02145)是最强证明**:用**冻结 SD VAE** 编解码**深度图**(非摄影、任意范围结构信号),只训 U-Net,重建良好;后续做 normal/edge/intrinsic 同法。**16 通道潜码是通用图像形状信号载体,不是只装照片。**
- **决定判断的 caveat**:Marigold 把深度**归一到 [−1,1] 再编码、解码后反归一**,从不让解码器吐越界值——**这正是 PU21/PQ 技巧**。VAE 能装**任何能压进其训练区间的信号,但不原生吐越界浮点**。恢复的动态范围受**压缩区间内解码器精度**限制,而非硬 [0,1] 墙。

### 完全跳出"LoRA + 感知编码"框架的方案

- **(a) 增益图分解**(GM-Diffusion ICCV25、GMODiff、Gain-MLP):优秀显示 HDR,但 ISO 增益图上限 ~+3–6 stop、显示构造,原生表达不了 10⁵ 太阳。中间件非 IBL 终目标。
- **(b) 两阶段 SDR 生成 → 专用 LDR2HDR/iTM 扩散精修器(条件于 SDR)**:干净分离"出好图"与"扩范围",精修器可训在真 EXR。但继承 iTM 天花板:裁剪高光幻觉、绝对太阳尺度靠猜。同一堵物理墙。
- **(c) 多曝光栈 + 合并**(LEDiff、Bracket Diffusion 2405.14304、DiffusionLight、2604.21008):领域**最可靠的冻结 VAE 扩范围法**(每张包围都在 VAE 舒适区)。**很可能是最有前途的替代第一步**——直接攻击冻结 VAE 精度问题而非硬扛。仍只 ~23×(~4.5 stop)融合,太阳级需多张包围 + 仔细高光处理。
- **(d) 参数化天空+太阳模型**(Lalonde-Matthews 11 参、Hošek-Wilkie、Prague Sky、LM-GAN 2302.00087、Physically-Based Sky-Modeling 2512.15632):**唯一按构造把太阳辐射搞对的一类**(拟合物理太阳+天空,太阳盘面承载正确绝对辐照度)。弱点:仅户外日光、空间丰富度低。**扩散出外观全景 + 参数化太阳补顶端几 stop 的混合,是通往真 IBL 级天空 HDRI 最可辩护的路径,很可能是你们没权衡过的。**

### 跨域"让窄域模型输出宽域数据"

RAW 生成(RawGen/RAW-Diffusion 2411.13150/RAW-Flow 2601.20364)是最近类比 = 冻结先验 + **微调 VAE 解码器**出场景线性。教训 + 警示:RawGen **仍裁到 [0,1]**、靠 DNG 元数据给绝对尺度。**通过重定用途的 VAE 吐无界浮点是脆弱的。** 去量化/位深扩展在未饱和区易、在裁剪顶端不适定。

### 技术顾问建议(区域 6,按回报排序)

1. **把"微调/扩容 VAE"规划为第 2 步**(冻结 VAE 是你的天花板,不是 LoRA)。
2. **认真评估包围分解(2604.21008/LEDiff)作替代第一步**,它绕过天花板 1 而不动 VAE。
3. **天空专门混合参数化太阳+天空拟合**(LM-GAN/2512.15632)——太阳按构造正确。
4. **辐射级验证而非感知级验证**:用生成的 EXR **照明一个漫反射+镜面探针**,与真值 IBL 比,查太阳是否承载正确的总辐照度占比。**这是整个子领域刻意不做的测试。**

### 意外发现 / 全新方向(区域 6,最高价值)

> **A. "小 LoRA + 感知编码"第一步在结构上注定只能得到显示级/伪 HDR,不是终点。** 这不是失败,是正确的廉价 PoC——但**从第一天就按"第一步"架构**,数据用真场景线性、别用 iTM 猜的高光当真值。

> **B. 太阳量级只能靠"多平面"或"参数化太阳"拿到,不是靠调编码曲线。** 三条真·IBL 路径(扩容 VAE / 曝光栈 / 参数化太阳)都把极端范围放进**单独平面/单独模型**。

> **C. LuxDiT + Sumit 的双表示/双头趋同是强信号**:连 NVIDIA 都不信任单次冻结 VAE 承载全范围。**若你们要真 scene-linear,终极形态大概率是"双潜表示(tonemap + linear/log)沿通道拼 + 轻量融合"**——这比"16→32 全量重训 DiT"更轻、且有 LuxDiT 现成范式。

> **D. 混合架构(很可能是你们最优但没想到的终局)**:扩散生成外观全景(冻结 VAE + LoRA,负责 ~14 stop 的"画面")+ 一个**参数化太阳/光源头**(负责顶端 3–6 stop 的绝对辐射)+ 增益图/曝光栈把两者缝合成场景线性 EXR。这样"画面好看"和"太阳算对"各由擅长的模块负责,避开了单平面精度墙。

> **E. 换基座 VAE 的机会**:Qwen-Image-VAE-2.0(2605.13565)已出;若 Krea 2 未来迁到新 VAE,值得重估其对压进的高动态编码值的往返上限——但这不改变"编码器不原生吐越界"的根本结论。

---

## 综合实施路线图(把 6 个区域缝成一条可执行路径)

### 阶段 0:半天,决定性往返实验(在写任何训练代码前)
- 取 ~200 张真 EXR(Poly Haven + Fairchild),分别用 **Log3G10 / LogC3 / asinh / PU21 / PQ** 做 `encode → Qwen VAE encode → decode → 反变换`,测**线性域 PSNR、仅高光 PSNR、每 stop 码值预算、banding 起始 EV**。
- **产出**:选定编码曲线 + 明确"冻结 VAE 在你目标峰值下够不够"。这一个实验决定后续所有分叉。

### 阶段 1:MVP —— 小 LoRA(1–2 周,验证"能出显示级真 HDR")
- **编码**:阶段 0 的赢家(默认 Log3G10 或对齐 X2HDR 的 PU21),固定 18%-gray 锚点,峰值压在 V≈0.85–0.9。
- **数据**:Poly Haven(全景→随机透视裁剪 + 保留全景子集)+ Fairchild;caption = tonemap→Qwen-VL + 注入 EV/太阳/峰值亮度元数据 + 触发词 `"scene-linear HDR"`。**Path A 数据量:数十–数百张即可起步。**
- **训练**:Krea 2 RAW 上 LoRA(rank 16–64,attn Q/K/V/O),rectified-flow + logit-normal t,**只用编码空间流匹配损失 + 适度潜空间高光加权**,保 min-SNR。VAE 全冻结。
- **迁移**:RAW LoRA → Turbo(官方路径,编码方案不破坏迁移)。
- **ComfyUI 落地**:VAEDecodeHDR(去 clamp)→ 反编码到线性节点 → SaveEXR(OIIO 32-bit float ZIP scene-linear)→ `oiiotool --stats` 往返断言。
- **验收**:不仅看 PU-PSNR/PSNR-µ,**还要渲染探针做辐射级验证**。预期:~10–14 stop 显示级 HDR;太阳会是"可信但未定标"。

### 阶段 2:按需升级(banding / 保真驱动)
- banding 不可接受 → 加 **LumaFlux 式单调 RQS 样条 post-decode 头** 或换 **asinh**;再不行 → **VAE conv-LoRA(方案 2.2)+ 潜统计正则**(监控 KL 防 Turbo 破坏)。
- 沉睡陷阱排查:核查 **decode 后噪声是否感知均匀**(log 空间的非均匀噪声)。

### 阶段 3:真 scene-linear / 太阳量级(架构升级,若产品需要 IBL 级)
- **优先双表示/双头(LuxDiT 范式 / Sumit 16→32 + DA-VAE 零初始化嫁接)**,或**曝光栈分解(LatentHDR/LEDiff)**,把极端范围放单独平面。
- **天空/户外**:叠**参数化太阳+天空拟合**(LM-GAN / Prague Sky),让太阳按构造辐射正确。
- Turbo 侧:Adapt-then-Distill 或 Uni-DAD 联合蒸馏(方案 2 的区域 2 讨论)。
- **验收**:探针照明 vs 真值 IBL,太阳承载正确总辐照度占比。

### 一句话决策树
> 冻结 VAE + 感知编码 = 免费拿显示级 HDR(第一步做这个);要 IBL 级太阳 = 必须多平面/参数化太阳(第三步);中间的 VAE conv-LoRA 是"想要更好保真但尽量不破坏 Turbo 迁移"的过渡档。

---

## 附录 A:编码曲线速查(公式 + 常数)

- **PU21(X2HDR 二次-log 形)**:`V=a(log2 L−Lmin)²+b(log2 L−Lmin)`;逆 `L=2^((2a·Lmin−b+√(b²+4aV))/(2a))`;a=0.001908,b=0.0078,Lmin=log2(0.005)。评测归一用 Pmax=595。完整 gfxdisp `banding_glare` 7 参:p=[0.353487901,0.3734658629,8.277e-05,0.9062562627,0.09150303166,0.9099517204,596.3148142]。
- **PQ/ST2084 逆**:`Np=N^(1/m2); L=((Np−c1)+/(c2−c3·Np))^(1/m1)·10000`;m1=0.1593017578125,m2=78.84375,c1=0.8359375,c2=18.8515625,c3=18.6875。
- **ARRI LogC3(EI800)**:`V=c·log10(aL+b)+d`(L>cut 否则 eL+f);cut=0.010591,a=5.555556,b=0.052272,c=0.247190,d=0.385537,e=5.367655,f=0.092809;逆 `L=(10^((V−d)/c)−b)/a`。天花板 ~55 线性。LogC4 天花板 ~470 线性。
- **RED Log3G10**:`V=0.224282·log10(155.975327·L+1)`(L≥0);`V=L/15.1927−0.01`(L<0);逆 `L=(10^(V/0.224282)−1)/155.975327`。18% 灰在 1/3,灰上 ~10 stop。
- **asinh/Lupton**:`V=asinh(L/β)/asinh(Lmax/β)`;逆 `L=β·sinh(V·asinh(Lmax/β))`;β=0.01–0.05。
- **DiffHDR Log-Gamma**:`T=(log(1+γx)/log(1+γM))^(1/γ)`;逆 `x=(exp(T^γ·log(1+γM))−1)/γ`。
- **µ-law(µ=255)**:`V=sign(x)ln(1+µ|x|)/ln(1+µ)`;逆 `|x|=((1+µ)^|V|−1)/µ`。
- **归一化铁律**:固定 18%-gray=0.18 锚点,绝不逐图 max;峰值压 V<1;逐图 scale 写 EXR 元数据。

## 附录 B:关键判断一览(可直接引用)

1. 路线 A(感知编码+冻结 VAE+LoRA)= 真实但显示级(~14 stop/4000 nit),X2HDR 已证。
2. 天花板是算术(~85 码给所有高光 ⇒ ~4–6 stop / 峰值 200–500),调参不能破。
3. Qwen VAE 解码器无 tanh、只有一行 clamp;瓶颈在编码器不在解码器。
4. 去 clamp ≠ HDR;必须前置范围压缩(log/PQ)。
5. 太阳量级只能多平面(扩容 VAE / 曝光栈 / 增益图)或参数化太阳。
6. 损失几乎不用设计:编码即损失;唯一值得加的是潜空间高光加权。
7. VAE decode 自带 clamp 是 ComfyUI 链路第一杀手(任务书漏了)。
8. S2R-HDR 是场景线性卫生源,非 IBL/太阳骨干(透视、未定标);真骨干 = Poly Haven + 程序化。
9. ITM 猜的太阳是毒药,绝不当真值;mask 裁剪光源或用参数化光。
10. 终极形态很可能是"双表示 + 参数化太阳"的混合,而非"16→32 全量重训"。

---

## 完整来源清单

**第一手核实(本报告作者亲验)**
- Krea 2 技术报告 https://www.krea.ai/blog/krea-2-technical-report · https://www.krea.ai/krea-2-open-source · diffusers krea2 pipeline https://huggingface.co/docs/diffusers/main/en/api/pipelines/krea2 · musubi-tuner krea2 https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md
- Qwen Image VAE 源码 https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py · 文档 https://huggingface.co/docs/diffusers/main/api/models/autoencoderkl_qwenimage · Qwen-Image-VAE-2.0 https://arxiv.org/abs/2605.13565
- X2HDR https://arxiv.org/abs/2602.04814 · Sumit Chatterjee https://sumitc.com/work/hdr-generation · Felldude https://huggingface.co/Felldude/Qwen-Image-HDR-VAE
- S2R-HDR https://huggingface.co/datasets/iimmortall/S2R-HDR · https://arxiv.org/abs/2504.07667 · https://openimaginglab.github.io/S2R-HDR/

**区域 1(编码)**:LumiVid https://arxiv.org/abs/2604.11788 · DiffHDR https://arxiv.org/abs/2604.06161 · LumaFlux https://arxiv.org/abs/2604.02787 · LatentHDR https://arxiv.org/abs/2605.11115 · LEDiff https://arxiv.org/abs/2412.14456 · PU21 https://github.com/gfxdisp/pu21 · Log3G10 https://antlerpost.com/colour-spaces/Log3G10.html · LogC4 spec https://www.arri.com/resource/blob/278790/dc29f7399c1dc9553d329e27f1409a89/2022-05-arri-logc4-specification-data.pdf · AgX https://github.com/sobotka/AgX · Khronos PBR Neutral https://github.com/KhronosGroup/ToneMapping · ISO 21496-1 https://www.iso.org/standard/86775.html · Ultra HDR https://developer.android.com/media/platform/hdr-image-format · Lupton asinh http://montage.ipac.caltech.edu/docs/Stretches/ · LTX-2 https://github.com/Lightricks/LTX-2 · LumiPic https://github.com/oumad/lumipic

**区域 2(VAE)**:GM-Diffusion https://github.com/Guanys-dar/GM-Diffusion · DA-VAE https://arxiv.org/abs/2603.22125 · Asymmetric VAE https://arxiv.org/pdf/2509.24142 · Conv-LoRA SR https://arxiv.org/pdf/2504.11271 · 压缩 VAE-LoRA https://arxiv.org/html/2606.16107 · FVAE-LoRA https://arxiv.org/pdf/2510.19640 · Uni-DAD https://arxiv.org/html/2511.18281 · Latent Upscale Adapter https://arxiv.org/pdf/2511.10629 · netocg/vae-decode-hdr https://github.com/netocg/vae-decode-hdr · Z-Image-Turbo de-distill https://lilting.ch/en/articles/z-image-turbo-lora-dedistill-adapter

**区域 3(数据)**:Poly Haven https://polyhaven.com/hdris · API https://api.polyhaven.com/assets?type=hdris · Open HDRI https://digitalproduction.com/2025/10/29/open-hdri-25-free-29k-hdris-more-to-come/ · Laval hdrdb http://hdrdb.com · Beyond the Pixel https://lvsn.github.io/beyondthepixel/ · Fairchild http://markfairchild.org/HDR.html · SingleHDR https://github.com/alex04072000/SingleHDR · HDRCNN https://computergraphics.on.liu.se/hdrcnn/ · DoRF/EMoR https://www.cs.columbia.edu/CAVE/software/softlib/dorf.php · Unprocessing https://people.csail.mit.edu/tfxue/papers/cvpr2019_unprocess.pdf · Deep HDR Hallucination https://pmc.ncbi.nlm.nih.gov/articles/PMC8230591/ · AIM 2025 ITM https://arxiv.org/html/2508.13479v1 · Blender Sky https://docs.blender.org/manual/en/latest/render/shader_nodes/textures/sky.html · Nishita disc bug https://developer.blender.org/T79249 · Infinigen https://infinigen.org/ · XRFeitoria https://github.com/openxrlab/xrfeitoria · Hypersim https://github.com/apple/ml-hypersim · LuxDiT https://research.nvidia.com/labs/toronto-ai/LuxDiT/ · Lighting in Motion https://arxiv.org/abs/2512.13597 · T-LoRA https://arxiv.org/abs/2507.05964

**区域 4(损失)**:RAW&HDR 训练 https://arxiv.org/pdf/2312.03640 · PU deep IQA https://arxiv.org/html/2405.00670v1 · ColorVideoVDP https://github.com/gfxdisp/ColorVideoVDP · FLIP https://github.com/NVlabs/flip · pyiqa https://github.com/chaofengc/IQA-PyTorch · HDR-VDP-3 https://arxiv.org/pdf/2304.13625 · NoR-VDPNet++ https://github.com/banterle/NoR-VDPNetpp · TMQI https://ece.uwaterloo.ca/~z70wang/research/tmqi/ · Cycle Ride to HDR https://arxiv.org/html/2410.15068v1 · Deep-HdrReconstruction https://github.com/marcelsan/Deep-HdrReconstruction · ExpoCM https://arxiv.org/pdf/2605.02464 · PhysHDR https://arxiv.org/html/2509.16869v1 · Min-SNR https://arxiv.org/html/2303.09556v3 · Improved Noise Schedule https://arxiv.org/html/2407.03297 · CompressedVQA-HDR https://arxiv.org/html/2507.11900

**区域 5(ComfyUI/EXR)**:HQ-Image-Save https://github.com/spacepxl/ComfyUI-HQ-Image-Save · ComfyUI nodes.py https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/nodes.py · comfy/sd.py https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/sd.py · SaveImageAdvanced PR https://github.com/Comfy-Org/ComfyUI/pull/13850 · ComfyUI-EXR-API https://github.com/nofunstudio/ComfyUI-EXR-API · CoCoTools_IO https://github.com/Conor-Collins/ComfyUI-CoCoTools_IO · ComfyUI_Gear https://github.com/oumad/ComfyUI_Gear · OpenEXR https://openexr.com/en/latest/TechnicalIntroduction.html · OpenImageIO https://openimageio.readthedocs.io/en/latest/imageoutput.html · oiiotool https://openimageio.readthedocs.io/en/latest/oiiotool.html · ACES/OCIO https://chrisbrejon.com/cg-cinematography/chapter-1-5-academy-color-encoding-system-aces/

**区域 6(新工作/替代)**:LatentHDR 2605.11115 · 曝光包围线性生成 https://arxiv.org/html/2604.21008 · GMODiff https://arxiv.org/pdf/2512.16357 · RawGen https://arxiv.org/html/2604.00093v1 · RAW-Diffusion https://arxiv.org/html/2411.13150v1 · RAW-Flow https://arxiv.org/pdf/2601.20364 · HDR env-map 潜扩散 https://arxiv.org/html/2507.21261v1 · LuxDiT https://arxiv.org/abs/2509.03680 · Marigold https://arxiv.org/html/2312.02145v2 · Gain-MLP https://arxiv.org/html/2503.11883v1 · DiffusionLight https://arxiv.org/abs/2312.09168 · Text2Light https://arxiv.org/pdf/2209.09898 · Deep HDR Hallucination https://arxiv.org/abs/2106.09486 · LM-GAN https://arxiv.org/pdf/2302.00087 · Physically-Based Sky-Modeling https://arxiv.org/html/2512.15632 · Bracket Diffusion https://arxiv.org/html/2405.14304v1

---

*报告完。6 个探索区域全部覆盖,含方案全景、可编码级实现、技术顾问 MVP/终极/顺序建议、辐射级验证方法,以及"第一步注定只能得显示级 HDR、真 IBL 级需多平面/参数化太阳"的直接判断。*
