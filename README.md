# Krea 2 HDR Toolkit

让 **Krea 2** 通过小 LoRA 生成**真·场景线性 HDR**(浮点、值 >> 1、可导出 EXR / IBL)的一套**独立工具与文档**:
编码曲线与 CPU 验证、数据集构建与核验、EXR / HDR-AVIF 转换、ComfyUI 节点包与工作流、以及完整报告(含 PDF)。

> **一句话结论**:"**LogC4 感知编码 + 冻结 VAE + 在 RAW 上训 DiT LoRA**" 能让 Krea 2 生成**真学到的**(非曲线编造的)扩展动态范围,并泛化到训练集外内容;ComfyUI 端到端可产出真 HDR EXR(峰值 459× / 8.7 stop)。单平面上限 ~470×(太阳级需多平面)。详见 `reports/`。

模型权重不在本仓库(见文末"权重与数据")。这里是**工具 + 报告**。

---

## 安装

```bash
pip install -r requirements.txt
# HDR-AVIF 导出还需系统工具:  apt-get install -y ffmpeg libavif-bin
# GPU 项(真 VAE 往返)可选:   pip install "torch>=2.4" "diffusers>=0.39"
```
> **依赖坑**:装 `opencv-python-headless`,别装 `opencv-python`(后者需 `libGL.so.1`)。

## 目录

```
krea2-hdr-toolkit/
├── tools/            所有 CPU 工具(扁平,import 互通;在此目录内运行)
├── comfyui/          ComfyUI 节点包 + 工作流 + LoRA 重映射
├── dataset/          数据集卡 + manifests(图像可由 build_dataset.py 重建)
├── reports/          完整报告(PDF + md)+ 图表包(figures/)
└── requirements.txt
```

## 核心概念(30 秒)

场景线性辐射(0 到几百×)用 **ARRI LogC4** 曲线压进 [0,1] → 存成 **8-bit PNG**(HDR 靠**编码**而非位深承载:log 曲线把 0–470× 铺在 256 级上,码 255 解码回 470×)→ 喂**冻结** Qwen Image VAE → 在 RAW 上训 DiT LoRA。推理时模型输出 LogC4 码,**inverse-LogC4** 展开回线性写 EXR。

```
训练:  EXR →(median→0.18 归一)→ LogC4[0,1] → 8-bit PNG → VAE latent → DiT LoRA(RAW)
推理:  prompt → Krea2+LoRA → LogC4[0,1] → inverse-LogC4 → 场景线性(>>1) → EXR / HDR AVIF
```

---

## 使用

所有 `tools/` 脚本在 `tools/` 目录内运行(相对路径 `data/`、`out/`)。

### 1) CPU 验证工具包(无需 GPU)
```bash
cd tools
python3 run_all.py                 # 一键跑:编码往返 + 精度 + EXR 链路 + 真实数据分析
# 或单独:
python3 test_precision.py          # 8 条曲线往返可逆(~1e-13)+ 8-bit 精度/banding + 图
python3 test_exr_roundtrip.py      # 32-bit float 保住 >1;复现 clamp/sRGB/half 三种"HDR杀手"
python3 test_vae_roundtrip.py      # (GPU)真 Qwen VAE 往返:各曲线码域 PSNR + 分带误差
python3 test_hlg_vs_logc4.py       # HLG vs LogC4 对比(结论:LogC4 更优,见下)
```

### 2) 编码曲线(`hdr_encodings.py`)
所有传递函数的正/反变换与注册表:`LogC4 / Log3G10 / LogC3 / PU21 / PQ / asinh / HLG / …`
```bash
python3 hdr_encodings.py           # 打印各曲线 code@0.18 与天花板
```
- 在 `CURVES` 里按名取 `(fwd, inv)`;`curve_ceiling(name)` 给 code=1 对应的线性峰值。
- **HLG vs LogC4(实测)**:LogC4 首选。HLG 无法同时"中灰 on-manifold"和"覆盖 11 stop"——中灰可用时天花板仅 12×(亮部裁剪),天花板拉到 470× 时中灰被压到近黑(off-manifold);LogC4 唯一两者兼得。(`test_hlg_vs_logc4.py`)

### 3) 数据集构建器与核验器
```bash
cd tools
python3 build_dataset.py --limit 40 --persp 8 --res 1024 --out ../dataset   # 从 Poly Haven CC0 重投影成透视场景图 + LogC4 编码
#   可 --limit 978 扩到全部 Poly Haven;输出 PNG + caption + manifest(可复现浮点裁剪)
python3 verify_hdr_content.py       # 逐图核验:训练图确实带可解码 HDR(应 ~93% 峰值>1)
```
- 训练用 `../dataset/manifest_hdr_rich.jsonl`(峰值>2 且 ≥3 stop 的子集)。
- 归一化:median→0.18 固定锚点(scale 写入 manifest)。

### 4) EXR / HDR-AVIF 转换工具(推理后处理)
```bash
cd tools
python3 decode_to_exr.py  gen.png  out.exr      # 生成的 LogC4 PNG → 场景线性 EXR(给渲染/IBL)
python3 to_display_hdr.py out.exr  out.avif     # EXR → HDR AVIF(PQ/BT.2020,给 HDR 屏;含 SDR 预览)
```
- `decode_to_exr`:inverse-LogC4 → 32-bit float EXR(无 clamp、无 sRGB、Rec.709 primaries)。也可读 LogC4 PNG。
- `to_display_hdr`:场景线性 → **保色相 rolloff + 高光去饱和** → PQ/BT.2020 → `avifenc` 出 10-bit HDR10 AVIF。
- **EXR 是主交付**(Blender/Nuke/Unreal,纹理设 linear/非 sRGB);AVIF 是显示端第二线。主档用 32-bit float + ZIP/PIZ 无损(half 会在 >65504 溢出;DWAA 劣化高光)。

### 5) ComfyUI(`comfyui/`)
```bash
# ① 重映射 LoRA(musubi 命名 → ComfyUI;不做则 0/264 不匹配、LoRA 无效)
cd comfyui/comfyui_krea_hdr
python3 remap_lora_for_comfyui.py in.safetensors out_COMFYUI.safetensors /path/to/ComfyUI /path/to/raw.safetensors
# ② 把 comfyui_krea_hdr/ 放进 ComfyUI/custom_nodes/;LoRA 放 models/loras/
# ③ 用 comfyui/workflow_krea2_hdr_api.json,或按 comfyui_krea_hdr/README.md 连图
```
两个节点:**LogC4ToLinear**(VAEDecode 后)+ **Save EXR Scene-Linear**(32-bit 无 clamp)。
> **关键坑**:**不要加 ModelSamplingSD3/shift 节点**——Krea2 已自带 shift,再加会双重 shift 过曝发白。绝不用内置 SaveImage 存 HDR(8-bit+clamp)。
>
> **上游进行中**:已给 ComfyUI 内置 `SaveImageAdvanced` 加了 `LogC4` colorspace 选项(inverse-LogC4→场景线性 EXR),合入后可省掉自定义保存节点。

### 6) 报告与图表(`reports/`)
- `Krea2-HDR-实施验证报告.pdf` —— 图文配对完整报告(真实曲线 + 表格 + 图像分析,8 页)。
- `implementation-report.md` / `research-findings.md` —— 实施与验证 / 路线调研。
- `figures/` —— **图表包**(figA–F 曲线 + 各分析图,`FigNN` 对应报告图号)。

---

## 核心结论(均有实测支撑)

1. 路线可行:LogC4 编码 + 冻结 VAE + RAW LoRA,Krea 2 学会生成扩展动态范围。
2. 真学到,非编造:同 prompt 同 seed,基座裁平高光(伪),LoRA 铺展高光(真),贴顶率 6.20%→0.66%。
3. 能泛化:训练集外 7/8 生成真 HDR,亮区落在光源上。
4. ComfyUI 可用:LoRA 需 key 重映射(0→264)+ 节点;e2e 产出 459×/8.7 stop EXR。
5. 诚实上限:单平面 ~470×,太阳级需多平面 / 增益图 / 参数化太阳。

## 权重与数据(不在本仓库)

- **LoRA**:私密 HF `LAXMAYDAY/krea2-scene-linear-hdr-lora`(含 musubi 版 + ComfyUI 重映射版)。
- **基座**:Krea-2-Raw、Qwen3-VL-4B、Qwen-Image VAE(各自 HF)。
- **数据集图像**:由 `build_dataset.py` 从 Poly Haven(CC0)重建。
- 完整训练/环境/复现细节见配套主仓库的 `HANDOVER.md`。

## License

代码 MIT(见 `LICENSE`)。数据源 Poly Haven 为 CC0。报告与图表随本仓库。
