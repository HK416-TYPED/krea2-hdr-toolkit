# Krea 2 → 场景线性 HDR:实施与验证报告

> 配套文档:
> - 《研究发现报告》`2026-07-08-krea2-hdr-research-findings.md`(路线调研、6 大探索区、判断)
> - 本报告:**实际实现、训练、测试、ComfyUI 落地的记录与结果**
> - 代码:`hdr_cpu_validation/`(CPU 验证 + 工具)、`comfyui_krea_hdr/`(ComfyUI 节点)、`training/`(数据集+训练)
> 日期:2026-07-08 · 环境:RTX PRO 6000 Blackwell 96GB / torch 2.9 cu130 / diffusers 0.39 / ComfyUI v0.27.0

---

## 0. 摘要

我们让 **Krea 2**(12.9B,Qwen Image VAE)通过一个**小 LoRA** 生成**真·场景线性 HDR**(可导出 EXR,值 >>1),并端到端验证到 ComfyUI 落地。

**核心结论(全部有实测数据支撑):**

1. **路线可行**:走"**LogC4 感知编码 + 冻结 VAE + 在 RAW 上训 DiT LoRA**"这条路,Krea 2 确实学会了生成扩展动态范围。
2. **是真学到,不是曲线编造**:同 prompt 同 seed 下,基座把高光**裁平贴顶**(伪 HDR),LoRA 把高光**铺展成有层次的高值**(真 HDR)。这是可测的差异(贴顶率 6.20% → 0.66%)。
3. **能泛化**:对训练集完全没有的内容(火/霓虹/人像/教堂/烛光),**7/8** 的 prompt 生成真 HDR,亮区精准落在光源上。
4. **ComfyUI 可用**:musubi 训的 LoRA **需要 key 重映射**才能加载(0→264),并且需要**两个自定义节点**(inverse-LogC4 + 无 clamp EXR 保存);已全部实现并端到端跑通(产出 peak 459× / 8.7 stop 的 EXR)。
5. **诚实上限**:单平面 LogC4 峰值封顶 **~470×**,达不到字面太阳量级(10⁵⁺);要真太阳需多平面/增益图/参数化太阳(见 §10)。质量受限于 epoch-10 小数据(40 张 HDRI),但**"HDR 真实性"与"画质"是两个独立的轴**,本项目证明了前者。

---

## 1. 方法与管线

**表示选择**:不改 VAE(保住官方 RAW→Turbo 迁移),把场景线性辐射用 **ARRI LogC4** 曲线压进 [0,1] 再喂冻结的 Qwen Image VAE;生成侧模型学习在 LogC4 空间输出;推理后 **inverse-LogC4** 展开回线性写 EXR。

```
训练:  scene-linear EXR ──median→0.18 归一──► LogC4 编码 [0,1] ──► 8-bit PNG ──► VAE latent ──► DiT LoRA(RAW)
推理:  prompt ──► Krea2 + HDR-LoRA ──► LogC4 码 [0,1] ──inverse-LogC4──► 场景线性(>>1) ──► EXR(32-bit) / HDR AVIF
```

**为什么 LogC4**:在真 Qwen VAE 上做往返测试,LogC4 的码域 PSNR 最高、亮部带最好(见 §2.4)。**关键性质**:VAEDecode 自带的 [0,1] clamp 对本方案无害——模型输出本就是 [0,1] 的 LogC4 码,>1 只在 inverse-LogC4 之后出现。

---

## 2. CPU 验证(无 GPU,`hdr_cpu_validation/`)

在写训练代码前,用纯数值方法验证"线性映射与还原"和"EXR 链路"。

### 2.1 编码往返可逆性(`test_precision.py`)
8 条曲线(Log3G10/LogC3/LogC4/PU21/PQ/asinh/Log-Gamma/µ-law)在工作范围内**往返误差 ~1e-13(float64)**——映射与反变换数值上精确,可安全放进管线。

### 2.2 中灰落点(on-manifold 性)
code@0.18:Log3G10 0.333、LogC3 0.391、LogC4 0.278、PU21 0.359、PQ 0.348(都贴近 sRGB 的 ~0.46);µ-law 0.015、Log-Gamma 0.809 严重错位 → 实测确认是坏选择。

### 2.3 精度天花板(对研究报告的修正)
均匀 8-bit 代理下,log 曲线给出**均匀 ~17 码/stop、天花板以下无 banding 墙**——所以研究报告里"~85 码 → 4–6 stop 算术天花板"**过于简化,被实测推翻**。真正限制是 VAE 的**非均匀**精度(码值偏向训练过的显示参考中间调),以及太阳溢出曲线天花板。

### 2.4 真·Qwen VAE 往返(`test_vae_roundtrip.py`,GPU)
用**真** Qwen Image VAE(z_dim 16;第一手确认解码器**无 tanh/sigmoid**,唯一边界是 `torch.clamp(-1,1)`)替换代理:

- **X2HDR 前提证实**:感知编码往返码域 PSNR ~35–38 dB、中+高光带误差 ~2%;而**朴素 scaled-linear 直接喂 VAE → 1–50 带误差 7211%**。这是"必须走感知编码"的铁证。
- **曲线排名(码域 PSNR,真实 HDRI)**:**LogC4 38.2** > PU21 37.7 > PQ 37.1 > Log3G10 36.9 > asinh 36.1 > LogC3 35.4。→ 选 **LogC4 主选,Log3G10 备选**;LogC3 天花板低(55)最早裁剪。
- **真 VAE 两端都劣化**(代理漏掉的发现):深阴影/暗部高频纹理 ~22–24%,太阳裁剪;干净窗口是中+高光(0.05–50)~2%。

### 2.5 EXR 链路(`test_exr_roundtrip.py`)
- 32-bit float **ZIP/PIZ/ZIPS 精确**保住 4000× 太阳;**DWAA 有损偏移 1.6%**;**half 在 >65504 溢出成 +Inf**。
- 复现两个"沉默杀手":**clamp→[0,1] 把 4000→1.0**(HDR 全丢);**隐式 sRGB 把 4000→33.4**(辐射被破坏)。
- `oiiotool --stats` 式二分:file_max < tensor_max ⇒ 保存节点裁了;tensor_max ≤ 1 ⇒ 模型没生成 HDR。

### 2.6 真实数据核实(`dataset_prep.py`)
- **Poly Haven `kloofendal_43d_clear`**:median 0.178(≈0.18 锚点✓)、**max 111,253×、DR 31.6 stop、含真太阳**。单平面 8-bit 还原:阴影/中调/高光/亮部误差全 **<1%**(到 ~500),但**太阳带 97–99% 误差**(溢出曲线天花板,仅 0.002% 像素)。
- **S2R-HDR 样本**:峰值仅 **21×、无太阳** → 只宜作**场景线性卫生源**,不是 IBL/太阳骨干。

### 2.7 HLG vs LogC4(实测,`test_hlg_vs_logc4.py`)
补做了 **HLG(BT.2100 Hybrid Log-Gamma)** 与 LogC4 的对比——结论:**LogC4 更适合本任务**。HLG 的曲线形状**无法同时**做到"中灰 on-manifold"和"覆盖 11 stop":

| 编码 | code@0.18 | 天花板 | 真 VAE 码域PSNR | mid 带线性误差 | 结论 |
|---|---|---|---|---|---|
| HLG(peak=12) | 0.21 | **12×** | 36.6 dB | 1.9% | 中灰尚可,但天花板太低,**亮部(50–500×)裁剪 88% 误差** |
| HLG(peak=470) | **0.034** | 470× | 50.9 dB* | **8.0%** | 天花板够,但中灰被压到近黑(off-manifold);*高 PSNR 是"暗图易重建"的假象 |
| **LogC4** | **0.278** | 470× | 38.2 dB | **2.4%** | **唯一同时满足**:on-manifold 中灰 + 满头顶空间 + 最均衡还原 |

**关键方法学提醒**:HLG(470) 的码域 PSNR(50.9)比 LogC4 高,但那是因为整图被压暗、码值 MSE 天然小——**看真实线性域的分带误差(mid 8% vs LogC4 2.4%)才是对的**。HLG 本质是**显示端 HDR** 曲线(纸白之上只几档高光),不是"把 11 stop 场景线性打包进 on-manifold 码"的相机 log。**LogC4 仍是首选。**

---

## 3. 数据集(`training/dataset/`,构建器 `hdr_cpu_validation/build_dataset.py`)

- **来源**:40 张 **Poly Haven CC0** HDRI(户外/室内/晴天多样)。等距柱状全景**重投影成正常透视场景图**
- **规模**:320 张透视图,其中 **229 张 HDR-rich**(`manifest_hdr_rich.jsonl`,峰值>2 且 ≥3 stop);均 LogC4 编码、8-bit PNG + `.txt` caption + `manifest.jsonl`(记录 encoding/anchor/exposure_scale/yaw/pitch/fov,可复现精确浮点裁剪)。
- **归一化**:median→0.18 固定锚点(记录 scale)。
- **caption**:元数据驱动(Poly Haven 类别/标签 + 触发短语 `scene-linear HDR ... high dynamic range, physically based lighting`)。

### 3.1 HDR 内容核实(`verify_hdr_content.py`)——关键
逐图确认训练图**真带可解码 HDR**,不是发灰的 SDR:
- **93% 的图** inverse-LogC4 后峰值 >1.0(真 HDR)。
- **同一 PNG 两种读法**:当 SDR 直读峰值 **1.0**(发灰扁平),正确 inverse-LogC4 解码峰值 **470×**(太阳/高光都在)。HDR 信息在文件里,只是被 log 编码了。
- 对源 EXR 校验:天花板以下误差 **0.8%**;太阳封顶 ~470(单平面极限)。

### 3.2 一个真实的数据缺陷
- **50% 的"with visible sun" caption 打在了画面里根本没有太阳的裁剪上**(错标)。
- **停车场类**:23 张裁剪只有 7 张画面含太阳,且都是"看向太阳"的天空裁剪;"看向地面"的停车场框取里太阳被裁到画面外。→ 模型把"停车场"(地面)和"太阳"(天空)学成了不共现的两件事,导致 prompt "停车场+太阳" 时太阳偏弱。
- **修复方向**:按裁剪真实内容诚实打标(测 hdr_peak 才写"sun")+ 宽 FOV 让太阳和场景共现。

---

## 4. 训练(`training/`)

- **框架**:musubi-tuner(kohya)`krea2_train_network.py`。
- **配置**:RAW 模型 + `networks.lora_krea2`,dim/alpha **32**(全 Linear 层,264 模块),lr 1e-4,adamw8bit,gradient checkpointing,`timestep_sampling shift discrete_flow_shift 2.5`,10 epoch / 2290 步,每 2 epoch 存点。
- **模型文件**:Krea-2-Raw `raw.safetensors`(25G)、Comfy-Org Qwen3-VL `qwen3vl_4b_bf16`(8.3G)、Qwen-Image VAE(243M)。
- **训练管线审计**:
  - **musubi ✅ 安全**:`krea2_cache_latents.py` 用 `/127.5-1`(= 我验证过的 [0,1]→[-1,1]),纯 RGB 读取,**无 sRGB/clamp/颜色增强**。
  - **ai-toolkit ⚠️**:归一化对,但自带 `ColorJitter` → **必须禁用一切颜色增强**(会在 LogC4 像素上乱改,毁掉编码)。只用几何增强(翻转)。
  - **通则**:LogC4 像素上不做任何颜色操作;Path A 下训练器**无需懂 HDR**(HDR 藏在编码里)。

---

## 5. 结果

### 5.1 端到端闭环(epoch 2 即证)
`prompt → Krea2 + HDR-LoRA → LogC4 → inverse-LogC4 → 场景线性 EXR`:峰值 **65×**、30% 像素 >1、太阳空间位置正确。

### 5.2 真伪对照(是学到,还是曲线编造)——核心证据
同 prompt 同 seed,基座 vs LoRA。判据是高光**贴顶率**(pinned@ceiling,越低=保留了真实层次):

| 场景 | 指标 | 基座 | HDR-LoRA |
|---|---|---|---|
| 室内窗 | 码值饱和率(code>0.98) | 7.88% | **1.09%** |
| 室内窗 | decode 后贴顶率 | 6.20% | **0.66%** |
| 夜景灯 | 码值饱和率 | 1.15% | **0.01%** |

**基座把窗户裁成一片死白贴顶(伪 HDR),LoRA 把窗户铺成有层次的高值(真 HDR)**。这是主干真学到了额外信息的直接证据。(注:仅"值>1"不能当证据——曲线会机械抬高任何图;证据是贴顶率下降 + 高光直方图从"贴顶尖峰"变成"铺开分布"。)

### 5.3 泛化(epoch 10,训练集范围外)
只挂 HDR 触发词,用社区常见 prompt(人像/壁炉/霓虹/龙焰/教堂彩窗/烛光):

- **7/8 生成真 HDR**:贴顶率 ≤0.01%,高光有层次,亮区精准落在真实光源上(霓虹牌/灯塔灯/彩窗/火焰)。
- epoch 8 失败的**烛光,epoch 10 修好**(黑帧 → 火焰 400×);壁炉 151→468×、龙焰 58→91×、教堂 131→198×。
- 第 8 个(spaceship-cockpit)**不是黑帧**——模型渲染了一个很暗的**深空星场**(SDR 提亮后可见星点 + 蓝色星云),但**没有强光源**,整场景码值仅 ~0.08,经 LogC4 解码后接近零线性(峰值 0.04×),故无 HDR 范围、开 HDR 后反而更暗看不清。属"暗场景 + 无亮源"的正常结果,非 HDR 机制失败;也暴露一个真实特性:**暗场景的低 LogC4 码解码后接近零线性,在 HDR 域会更不可见**(可用推理时曝光归一 + 更多暗场景数据缓解)。

### 5.4 色彩核实
高光**中性、不偏蓝**(灯塔灯 R:G:B=0.96:1.00:0.74)。此前观察到的"蓝色溢出"是**显示端**问题(逐通道硬裁、无高光去饱和),已修(见 §6),**EXR 主产物不受影响**。

---

## 6. 显示交付(`decode_to_exr.py` / `to_display_hdr.py`)

- **主交付线 = 场景线性 EXR**(给渲染/IBL:Blender/Nuke/Unreal,纹理设 linear/非 sRGB)。
- **第二线 = HDR AVIF**(给 HDR 屏):scene-linear → **保色相软 rolloff + 高光去饱和** → PQ/BT.2020 → `avifenc` 出 10-bit HDR10 AVIF(已核验标签正确)。
  - **修正**:早期用"逐通道硬裁到 1000 nit、无去饱和",饱和蓝天/霓虹在 HDR 屏上"电蓝溢出";改为保色相 rolloff + 高光向白过渡后消除。
- **半浮点警示**:太阳可能 >65504 → half 溢出 +Inf;主档用 **32-bit float**。压缩用 **ZIP/PIZ 无损**;**DWAA 有损会劣化高光**,避免用于主档。

---

## 7. ComfyUI 落地(`comfyui_krea_hdr/`)——端到端实测通过

### 7.1 LoRA 能否直接加载?——不能,需 key 重映射
- 用 ComfyUI 自己的加载器实测:musubi LoRA **0/264 模块匹配**(musubi 用原生层名 `lora_unet_blocks_0_attn_wq`,ComfyUI 期望 diffusers 派生 key)。
- 提供 `remap_lora_for_comfyui.py`:**264/264 全部映射**;ComfyUI 启动日志确认 **"Model Krea2 … 263 patches attached"**(LoRA 真挂上)。
- 产物:`output/krea2-hdr-logc4-v1-COMFYUI.safetensors`。

### 7.2 EXR 保存节点适用吗?——内置不行,需两个自定义节点
- 内置 **SaveImage**:8-bit + clamp → HDR 全丢,禁用。
- 缺口:ComfyUI 无 inverse-LogC4 节点。提供节点包 `comfyui_krea_hdr`:
  - **`LogC4 to Linear`**:LogC4 码 → 场景线性(>1),插在 VAEDecode 后。
  - **`Save EXR Scene-Linear`**:32-bit 浮点、无 clamp、无 sRGB、标记 scene-linear(OpenEXR)。
  - 也可用现成 HQ-Image-Save(`tonemap=linear`),但**前面必须接 LogC4→Linear**。

### 7.3 端到端实测
原生 ComfyUI 图:`UNETLoader → CLIPLoader(krea2) → VAELoader → LoraLoader(重映射) → CLIPTextEncode → EmptySD3LatentImage → KSampler → VAEDecode → LogC4ToLinear → SaveEXRSceneLinear`。
- 产出真·场景线性 HDR EXR:**峰值 459×、8.7 stop、太阳在地平线**(质量与 musubi 一致)。
- **关键坑**:**不要加 `ModelSamplingSD3`/shift 节点**。Krea2 已自带分辨率相关 shift,再加会**双重 shift** → 画面过曝发白(code_median 0.80、仅 2.8 stop);去掉后正常(0.36、8.7 stop)。
- 工作流文件:`comfyui_krea_hdr/workflow_krea2_hdr_api.json`。

---

## 8. 结论

- **"小 LoRA + 感知编码 + 冻结 VAE"这条第一步路线成立**:Krea 2 学会了生成真·扩展动态范围(非曲线编造),并泛化到任意内容。
- **全链路已打通并在 ComfyUI 落地**:研究 → CPU 验证 → 数据集(核实带真 HDR)→ 训练管线审计 → LoRA 训练(epoch 10)→ 生成 → 泛化验证 → 显示修正 → ComfyUI 端到端(含 LoRA 重映射 + 自定义 EXR 节点)。

## 9. 局限

- **单平面上限 ~470×**:达不到字面太阳量级(10⁵⁺);太阳盘面会封顶。
- **画质粗糙**:epoch 10、40 张 HDRI 的小数据;质量是独立的轴,靠更多训练/更好数据提升。
- **数据缺陷**:50% "sun" 错标 + 裁剪让"场景⟂太阳"(§3.2)。
- **Turbo 迁移未测**:本次只训/测 RAW;RAW→Turbo LoRA 迁移(编码方案不改潜空间,理论上兼容)待验证。

## 10. 后续方向(按性价比)

1. **数据(最省)**:按裁剪真实内容诚实打标 + 宽 FOV 让太阳和场景共现;扩到全部 978 张 Poly Haven + Fairchild(公有领域)+ 程序化 Blender/Nishita(显式 Sun lamp 保 10⁴–10⁹)。
2. **要太阳级真实**:多平面/增益图(HDR=SDR·2^gain)或 LatentHDR 曝光栈,把极端范围放单独平面。
3. **天空 IBL**:叠参数化太阳+天空模型(LM-GAN/Prague Sky),让太阳按物理正确。
4. **表示层根治编造**:训练 VAE 编码器(16→32,Sumit 式)——代价是破坏 Turbo 免费迁移(见研究报告路线 B)。
5. **RAW→Turbo 迁移**:把 RAW 训的 LoRA 应用到 Turbo 验证快速推理。

---

## 附录 A:交付物清单

| 路径 | 内容 |
|---|---|
| `2026-07-08-krea2-hdr-research-findings.md` | 研究发现报告(路线调研、6 探索区、判断) |
| `2026-07-08-krea2-hdr-implementation-report.md` | 本报告 |
| `hdr_cpu_validation/` | CPU 验证套件:`hdr_encodings.py`、`test_precision.py`、`test_exr_roundtrip.py`、`test_vae_roundtrip.py`、`dataset_prep.py`、`build_dataset.py`、`verify_hdr_content.py`、`decode_to_exr.py`、`to_display_hdr.py`、`run_all.py`、`README.md`、`out/`(图表+EXR+AVIF) |
| `training/dataset/` | 320 张 LogC4 训练图 + caption + `manifest.jsonl` / `manifest_hdr_rich.jsonl` + `DATASET_CARD.md` |
| `training/output/krea2-hdr-logc4-v1.safetensors` | 最终 LoRA(epoch 10);另有 epoch 2/4/6/8 |
| `training/output/krea2-hdr-logc4-v1-COMFYUI.safetensors` | **ComfyUI 用**(已重映射 key) |
| `comfyui_krea_hdr/` | ComfyUI 节点包:`__init__.py`(LogC4ToLinear + SaveEXRSceneLinear)、`remap_lora_for_comfyui.py`、`workflow_krea2_hdr_api.json`、`README.md`、`TRAINING_NOTES.md`(在 `training/`) |
| HDR 画廊(artifact) | https://claude.ai/code/artifact/661366fe-476f-45fb-804d-b44174f8b350 |

## 附录 B:关键数字一览

- 编码往返误差:~1e-13(float64,8 曲线)
- 真 VAE 码域 PSNR:LogC4 38.2 dB(最优);朴素线性高光误差 7211%
- 真实 HDRI:kloofendal max 111,253×、31.6 stop、median 0.178;S2R-HDR 峰值 21×(无太阳)
- 数据集:320 图 / 229 HDR-rich;93% 解码为真 HDR;50% "sun" 错标
- 训练:264 LoRA 模块、dim/alpha 32、2290 步、10 epoch
- 真伪对照(窗):贴顶率 基座 6.20% → LoRA 0.66%;饱和率 7.88% → 1.09%
- 泛化:7/8 OOD 真 HDR,贴顶率 ≤0.01%
- ComfyUI:LoRA 匹配 0→264(重映射后),263 patches attached;e2e EXR 峰值 459×、8.7 stop
- 单平面上限:~470×(LogC4 天花板)

## 附录 C:关键命令

```bash
# CPU 验证
cd hdr_cpu_validation && python3 run_all.py
# 建数据集(可 --limit 978 扩展)
python3 build_dataset.py --limit 40 --persp 8 --out ../training/dataset
# 核实 HDR 内容
python3 verify_hdr_content.py
# 训练(musubi)
cd training/musubi-tuner && python3 src/musubi_tuner/krea2_cache_latents.py --dataset_config ../config/hdr_lora.toml --vae <vae>
python3 src/musubi_tuner/krea2_cache_text_encoder_outputs.py --dataset_config ../config/hdr_lora.toml --text_encoder <te>
accelerate launch ... krea2_train_network.py --network_module networks.lora_krea2 --network_dim 32 --network_alpha 32 ...
# ComfyUI 用:重映射 LoRA
cd comfyui_krea_hdr && python3 remap_lora_for_comfyui.py in.safetensors out_COMFYUI.safetensors /path/to/ComfyUI /path/to/raw.safetensors
# 生成结果 → EXR / HDR AVIF
python3 hdr_cpu_validation/decode_to_exr.py gen.png out.exr
python3 hdr_cpu_validation/to_display_hdr.py out.exr out.avif
```

---
*报告完 · 2026-07-08*
