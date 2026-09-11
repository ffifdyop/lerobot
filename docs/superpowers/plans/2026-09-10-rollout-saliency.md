# Rollout Saliency 叠加视频 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SmolVLA rollout 期间，每个相机每个 policy tick 生成一张 saliency map（gradients×input，严格对应实际执行的 action），叠加到原始相机帧上，写入每相机一个 mp4 并实时推送到 rerun。

**Architecture:** 控制线程（GPU 0，永远 no_grad）在 RTC 推理前自己采样噪声，传给 `predict_action_chunk(noise=...)` 并把同一噪声 + 预处理观测 + chunk kwargs 扔进容量为 1 的旁路队列（丢旧留新，绝不阻塞控制循环）。旁路线程持有加载在第二块 GPU 上的模型副本，用同一噪声重放 forward（开梯度，只对输入图像张量求梯度），backward 得到每像素 saliency，渲染成 turbo 热力图叠加帧后写视频 + 推 rerun。Explainer 按策略类型注册，未注册的策略在启动时 fail-fast。

**Tech Stack:** PyTorch autograd（输入梯度，参数冻结）· OpenCV（VideoWriter / COLORMAP_TURBO / GaussianBlur）· rerun-sdk（复用现有 viz 后端）· draccus dataclass 配置 · threading.Queue(maxsize=1)。

**Key facts (已核实，实现时直接依赖):**

- SmolVLA 推理入口 `predict_action_chunk(batch, noise=None, **kwargs)` 已接受噪声注入参数：`src/lerobot/policies/smolvla/modeling_smolvla.py:231-241`。`noise=None` 时内部 `sample_noise`（无 seed）：`src/lerobot/policies/common/flow_matching.py:43-51`。
- 传入相同 noise + 相同观测 + 相同 `inference_delay`/`prev_chunk_left_over` 可**确定性重放**实际执行的 chunk（同输入→同输出；跨 GPU 型号的浮点核实现可能有 ulp 级差异，对热力图无可感知影响）：`RTCProcessor` 无内部可变状态（只有 debug tracker），见 `src/lerobot/policies/rtc/modeling_rtc.py:122-254`。
- **运行时 RTC 配置注入在策略加载之后**：`policy.config.rtc_config = cfg.inference.rtc` + `init_rtc_processor()` 在 `src/lerobot/rollout/context.py:338-341`，且 `from_pretrained` **共享 config 对象**（`src/lerobot/policies/pretrained.py:97`）。lerobot-train 保存的 checkpoint `rtc_config=None`（`configuration_smolvla.py:104`）→ **SaliencyManager 必须创建于该注入之后**，否则 explainer 副本 `rtc_processor=None` 但 config 显示 RTC 开启，重放每帧 `AttributeError`（评审 C3）。
- `rollout.sh:5` 有 `export CUDA_VISIBLE_DEVICES=2` —— 进程内只见一张卡，`resolve_explainer_device` 的"第二张 GPU"逻辑不会触发；启用 saliency 时需把它改成多卡掩码（评审 I3，见 Task 7/8）。
- 图像→动作全链路可微（`resize_with_pad`/dtype cast 无 detach）：`modeling_smolvla.py:334-380`、`smolvlm_with_expert.py:202-214`。
- `n_obs_steps=1`（`configuration_smolvla.py:28`）→ 模型副本无需历史队列即可精确重放。
- RTC 引擎的推理调用点：`src/lerobot/rollout/inference/rtc.py:503-505`；噪声 shape 用 `(1, config.chunk_size, config.max_action_dim)`（`modeling_smolvla.py:740-742`）。
- base strategy 无 episode 概念：一次 run = 一个视频文件（`strategies/base.py:61`）。
- rerun 后端：`src/lerobot/utils/rerun_visualization.py`；策略副本加载复用 `src/lerobot/rollout/context.py:271` 的 `_load_pretrained_policy`。
- 测试 mock 机器人配置：`tests/mocks/mock_robot.py` 的 `MockRobotConfig`（`tests/test_rollout.py:242` 的先例）。

**Out of scope（共识已定）:** DM05/openpi 远程路径（本地无梯度）；per-dim 热力图；integrated gradients（接口预留 `method` 字段但只实现 `grad_x_input`）。

---

## File Structure

```
src/lerobot/rollout/saliency/          # 新包
├── __init__.py                       # 轻量导出 SaliencyConfig/Explainer/SmolVLAExplainer（触发注册）；manager 一律经子模块引用
├── config.py                         # SaliencyConfig dataclass + validate_saliency()
├── overlay.py                        # 纯函数：tensor→uint8 BGR、min-max 归一化、turbo 叠加
├── explainer.py                      # Explainer 基类 + 注册表 + get_explainer_cls fail-fast
├── smolvla.py                        # SmolVLAExplainer：梯度重放适配器
├── recorder.py                       # SaliencyRecorder：旁路线程 + cv2 写视频 + rerun 推流
└── manager.py                        # SaliencyManager：选 GPU、加载模型副本、组装 lifecycle

修改:
├── src/lerobot/rollout/configs.py        # saliency 字段 + __post_init__ 校验
├── src/lerobot/rollout/inference/rtc.py  # saliency_sink 属性 + 噪声注入 + 快照
├── src/lerobot/rollout/context.py        # 创建 manager、挂 sink、PolicyContext 字段
├── src/lerobot/scripts/lerobot_rollout.py# finally 中 stop manager
├── src/lerobot/utils/rerun_visualization.py # log_rerun_saliency()
└── rollout.sh                            # 注释掉的启用示例

测试: tests/test_saliency.py（全部新测试集中于此，与 tests/test_rollout.py 同级）
```

---

### Task 1: SaliencyConfig + 校验 + RolloutConfig 接线

**Files:**
- Create: `src/lerobot/rollout/saliency/__init__.py`（先放最小内容，Task 3 再补导出；保持轻量——不得引入 manager/recorder，理由见 Task 6 评审 M2 说明）
- Create: `src/lerobot/rollout/saliency/config.py`
- Modify: `src/lerobot/rollout/configs.py:222-293`（RolloutConfig 加字段）、`configs.py:336-337` 之后（__post_init__ 加校验）
- Test: `tests/test_saliency.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_saliency.py`。

> **import 纪律（评审 I5）**：本文件的全部 import 一律放在文件顶部 import 区（ruff 全局启用 E 规则，tests 不豁免 E402）。后续任务追加测试时，把新 import **插入顶部 import 区**，绝不写在文件中部。

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for rollout-time saliency (explainer, recorder, config plumbing)."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

# ---------------------------------------------------------------------------
# SaliencyConfig validation
# ---------------------------------------------------------------------------


def test_validate_saliency_requires_rtc_engine():
    from lerobot.rollout.inference import SyncInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    with pytest.raises(ValueError, match="inference.type=rtc"):
        validate_saliency(SaliencyConfig(enabled=True), SyncInferenceConfig(), "rerun", True)


def test_validate_saliency_rejects_unknown_method():
    from lerobot.rollout.inference import RTCInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    cfg = SaliencyConfig(enabled=True, method="integrated")
    with pytest.raises(ValueError, match="grad_x_input"):
        validate_saliency(cfg, RTCInferenceConfig(), "rerun", True)


def test_validate_saliency_requires_rerun_for_display():
    from lerobot.rollout.inference import RTCInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    with pytest.raises(ValueError, match="rerun"):
        validate_saliency(SaliencyConfig(enabled=True), RTCInferenceConfig(), "foxglove", True)


def test_validate_saliency_requires_display_data_for_display():
    from lerobot.rollout.inference import RTCInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    # display_data=False but saliency.display=True → rerun never initialized, silent no-op
    with pytest.raises(ValueError, match="--display_data=true"):
        validate_saliency(SaliencyConfig(enabled=True), RTCInferenceConfig(), "rerun", False)


def test_validate_saliency_warns_on_torch_compile(caplog):
    import logging as _logging

    from lerobot.rollout.inference import RTCInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    with caplog.at_level(_logging.WARNING):
        validate_saliency(SaliencyConfig(enabled=True), RTCInferenceConfig(), "rerun", True, use_torch_compile=True)
    assert any("torch.compile" in r.message for r in caplog.records)


def test_validate_saliency_passes_when_disabled():
    from lerobot.rollout.inference import SyncInferenceConfig
    from lerobot.rollout.saliency.config import SaliencyConfig, validate_saliency

    validate_saliency(SaliencyConfig(enabled=False), SyncInferenceConfig(), "foxglove", False)


def test_rollout_config_wires_saliency_validation():
    from lerobot.rollout import RolloutConfig
    from lerobot.rollout.saliency.config import SaliencyConfig
    from tests.mocks.mock_robot import MockRobotConfig

    # default inference is sync → enabled saliency must fail fast, before policy loading
    with pytest.raises(ValueError, match="inference.type=rtc"):
        RolloutConfig(robot=MockRobotConfig(), saliency=SaliencyConfig(enabled=True))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_saliency.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lerobot.rollout.saliency'`

- [ ] **Step 3: Write minimal implementation**

创建 `src/lerobot/rollout/saliency/config.py`：

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Configuration for rollout-time saliency visualization (see CONTEXT.md: Saliency Map / Overlay Video / Explainer)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SaliencyConfig:
    """Saliency overlay video settings (``--saliency.*`` flags on lerobot-rollout)."""

    # Master switch. Requires a local policy with a registered Explainer and the RTC inference engine.
    enabled: bool = False
    # Saliency method. Only "grad_x_input" is implemented today; "integrated" is a planned upgrade.
    method: str = "grad_x_input"
    # Stream overlay frames to rerun entities ``saliency/<camera>`` (requires --display_data).
    display: bool = True
    # Device for the explainer's model copy. None = auto: a second GPU if available, else the policy device.
    device: str | None = None
    # Output directory for the per-camera mp4s. None = outputs/saliency/<run timestamp>/
    output_dir: str | None = None
    # Fixed fps of the saved mp4s (frames arrive at policy tick rate; dropped frames just skip).
    video_fps: float = 10.0
    # Blend weight of the heatmap in the overlay.
    alpha: float = 0.5
    # Gaussian smoothing sigma (px at camera resolution) applied to the raw saliency map. 0 disables.
    blur_sigma: float = 2.0


def validate_saliency(
    saliency: SaliencyConfig,
    inference,
    display_mode: str,
    display_data: bool,
    use_torch_compile: bool = False,
) -> None:
    """Fail fast on saliency flag combinations that cannot work. No-op when disabled."""
    if not saliency.enabled:
        return
    import logging

    from lerobot.rollout.inference import RTCInferenceConfig

    logger = logging.getLogger(__name__)
    if not isinstance(inference, RTCInferenceConfig):
        raise ValueError(
            "--saliency.enabled=true currently requires --inference.type=rtc "
            "(noise injection lives in the RTC engine)"
        )
    if saliency.method != "grad_x_input":
        raise ValueError(
            f"Unsupported --saliency.method={saliency.method!r} (only 'grad_x_input' is implemented)"
        )
    if saliency.display and display_mode != "rerun":
        raise ValueError("--saliency.display=true requires --display_mode=rerun")
    if saliency.display and not display_data:
        raise ValueError(
            "--saliency.display=true requires --display_data=true (rerun is never initialized otherwise)"
        )
    if not 0.0 < saliency.alpha < 1.0:
        raise ValueError(f"--saliency.alpha must be in (0, 1), got {saliency.alpha}")
    if use_torch_compile:
        # Not fatal: compiled inductor graphs may reorder FP ops, so the eager replay is
        # approximate rather than deterministic. Warn loudly instead of failing.
        logger.warning(
            "--saliency.enabled=true combined with --use_torch_compile: the explainer replays "
            "eagerly, so heatmaps may drift slightly from the compiled executed action"
        )
```

创建 `src/lerobot/rollout/saliency/__init__.py`（本任务最小内容）：

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Rollout-time saliency: per-camera input-pixel importance maps of the executed action."""

from .config import SaliencyConfig
```

修改 `src/lerobot/rollout/configs.py`：

1. 顶部 import 区（`from .inference import ...` 附近）加：
```python
from .saliency.config import SaliencyConfig
from .saliency.config import validate_saliency
```
2. `RolloutConfig` 字段区，`display_compressed_images`（line 276）之后加：
```python
    # Saliency overlay videos (--saliency.*). See rollout/saliency/config.py.
    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)
```
3. `__post_init__` 中，`autosteer_interval_s` 校验（line 337）之后、Sentry streaming 段之前插入：
```python
        validate_saliency(
            self.saliency, self.inference, self.display_mode, self.display_data, self.use_torch_compile
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_saliency.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/rollout/saliency/__init__.py src/lerobot/rollout/saliency/config.py src/lerobot/rollout/configs.py tests/test_saliency.py
git commit -m "feat(saliency): add SaliencyConfig with fail-fast validation wired into RolloutConfig"
```

---

### Task 2: Overlay 渲染纯函数

**Files:**
- Create: `src/lerobot/rollout/saliency/overlay.py`
- Test: `tests/test_saliency.py`（追加）

- [ ] **Step 1: Write the failing test**

追加到 `tests/test_saliency.py`：

```python
# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------


def test_normalize_map_min_max():
    from lerobot.rollout.saliency.overlay import normalize_map

    m = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    n = normalize_map(m)
    assert n.min() == 0.0 and n.max() == 1.0


def test_render_overlay_shape_and_dtype():
    from lerobot.rollout.saliency.overlay import render_overlay

    img = np.full((100, 180, 3), 128, dtype=np.uint8)
    sal = np.random.rand(100, 180).astype(np.float32)
    out = render_overlay(img, sal)
    assert out.shape == (100, 180, 3)
    assert out.dtype == np.uint8


def test_render_overlay_resizes_mismatched_map():
    from lerobot.rollout.saliency.overlay import render_overlay

    img = np.full((50, 60, 3), 200, dtype=np.uint8)
    sal = np.zeros((25, 30), dtype=np.float32)
    out = render_overlay(img, sal, blur_sigma=0.0)
    assert out.shape == (50, 60, 3)


def test_render_overlay_blur_flattens_singleton_map():
    """A uniform map must stay uniform after blur+normalize (constant map → colormap index 0, dark blue)."""
    from lerobot.rollout.saliency.overlay import render_overlay

    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    sal = np.full((32, 32), 7.0, dtype=np.float32)
    out = render_overlay(img, sal)
    # every pixel blended identically → rows identical
    assert (out == out[0]).all()


def test_render_overlay_alpha_zero_is_identity():
    from lerobot.rollout.saliency.overlay import render_overlay

    img = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
    sal = np.random.rand(20, 30).astype(np.float32)
    out = render_overlay(img, sal, alpha=0.0, blur_sigma=0.0)
    assert (out == img).all()


def test_tensor_to_uint8_bgr_handles_batch_and_rgb():
    from lerobot.rollout.saliency.overlay import tensor_to_uint8_bgr

    t = torch.zeros(1, 3, 8, 6)
    t[:, 0] = 1.0  # RGB red
    arr = tensor_to_uint8_bgr(t)
    assert arr.shape == (8, 6, 3)
    assert arr.dtype == np.uint8
    assert arr[0, 0, 2] == 255  # red lands at BGR index 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_saliency.py -v -k overlay`
Expected: FAIL — `ModuleNotFoundError: No module named 'lerobot.rollout.saliency.overlay'`

- [ ] **Step 3: Write minimal implementation**

创建 `src/lerobot/rollout/saliency/overlay.py`：

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Overlay rendering: saliency map + raw camera frame → blended uint8 BGR frame."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def tensor_to_uint8_bgr(img: torch.Tensor) -> np.ndarray:
    """(B,C,H,W) or (C,H,W) image tensor (float [0,1] or uint8, RGB) → (H,W,3) uint8 BGR."""
    if img.ndim == 4:
        img = img[0]
    arr = img.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).round().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # CHW (RGB) → HWC (RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def normalize_map(m: np.ndarray) -> np.ndarray:
    """Per-frame min-max normalization to [0, 1]."""
    lo = float(m.min())
    hi = float(m.max())
    return (m - lo) / (hi - lo + 1e-8)


def render_overlay(
    img_bgr: np.ndarray,
    saliency: np.ndarray,
    alpha: float = 0.5,
    blur_sigma: float = 2.0,
) -> np.ndarray:
    """Blend a per-frame-normalized turbo heatmap onto a uint8 BGR frame."""
    if saliency.shape != img_bgr.shape[:2]:
        saliency = cv2.resize(saliency, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    if blur_sigma and blur_sigma > 0:
        ksize = 2 * round(3 * blur_sigma) + 1
        saliency = cv2.GaussianBlur(saliency, (ksize, ksize), blur_sigma)
    norm = normalize_map(saliency)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat, alpha, 0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_saliency.py -v`
Expected: 12 PASS

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/rollout/saliency/overlay.py tests/test_saliency.py
git commit -m "feat(saliency): add overlay rendering helpers (turbo colormap, min-max, blur)"
```

---

### Task 3: Explainer 注册表 + SmolVLA 梯度重放适配器

**Files:**
- Create: `src/lerobot/rollout/saliency/explainer.py`
- Create: `src/lerobot/rollout/saliency/smolvla.py`
- Modify: `src/lerobot/rollout/saliency/__init__.py`
- Test: `tests/test_saliency.py`（追加）

说明：适配器复刻 `modeling_smolvla.py:225-241`（`_prepare_batch` → `populate_queues` → stack → `prepare_images/prepare_state` → `model.sample_actions` → 裁剪到 `action_feature.shape[0]` → pi_aloha 编码）的调用序列，但不开 no_grad，并在 stack 之后给图像张量设 `requires_grad`。backward 目标是**实际执行的第一个 action**（`actions[0, 0]`）12 维的平方 L2（与 L2 范数只差一个标量因子，被逐帧 min-max 归一化吸收）。单元测试用 FakeSmolVLAPolicy 复刻同一内部接口（真模型重放保真度由 Task 8 的上机 E2E 验证——repo 内没有可在 CI 里构造完整 SmolVLA batch 的先例，见 `tests/policies/smolvla/test_smolvla_rtc.py:92` 的 skipif 先例）。

- [ ] **Step 1: Write the failing test**

追加到 `tests/test_saliency.py`（不新增 import——`deque`/`SimpleNamespace`/constants 已在 Task 1 的文件顶部）：

```python
# ---------------------------------------------------------------------------
# Explainer registry + SmolVLA grad-replay adapter (FakeSmolVLAPolicy stand-in)
# ---------------------------------------------------------------------------

CAM_KEY = "observation.images.cam1"


class _FakeFlowMatching(torch.nn.Module):
    """Tiny differentiable stand-in for VLAFlowMatching.sample_actions."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.head = torch.nn.Linear(8, action_dim)
        self.last_kwargs: dict | None = None

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        self.last_kwargs = {"noise": noise, **kwargs}
        x = self.conv(images[0]).mean(dim=(2, 3))
        first = self.head(x)  # (B, action_dim)
        return first.unsqueeze(1).repeat(1, 10, 1)  # (B, chunk, action_dim)


class _FakeSmolVLAPolicy(torch.nn.Module):
    """Stand-in exposing the SmolVLAPolicy internals the explainer touches.

    Must be an nn.Module: the adapter calls ``policy.eval()`` and derives the
    device from ``next(policy.parameters())`` (评审 C1).
    """

    def __init__(self, action_dim: int = 4):
        super().__init__()
        self.config = SimpleNamespace(
            image_features=[CAM_KEY],
            action_feature=SimpleNamespace(shape=(action_dim,)),
            adapt_to_pi_aloha=False,
        )
        self.model = _FakeFlowMatching(action_dim)  # submodule → parameters()/device work
        self._queues = {
            CAM_KEY: deque(maxlen=1),
            OBS_STATE: deque(maxlen=1),
            OBS_LANGUAGE_TOKENS: deque(maxlen=1),
            OBS_LANGUAGE_ATTENTION_MASK: deque(maxlen=1),
        }

    def _prepare_batch(self, batch):
        return batch

    def prepare_images(self, batch):
        images, masks = [], []
        for key in self.config.image_features:
            img = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            images.append(img)
            masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=img.device))
        return images, masks

    def prepare_state(self, batch):
        return batch[OBS_STATE]


def _make_fake_batch(h: int = 32, w: int = 32) -> dict:
    return {
        CAM_KEY: torch.rand(1, 3, h, w),
        OBS_STATE: torch.rand(1, 6),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.long),
    }


def test_get_explainer_cls_fails_fast_for_unregistered_policy():
    from lerobot.rollout.saliency.explainer import get_explainer_cls

    with pytest.raises(ValueError, match="No saliency explainer registered"):
        get_explainer_cls("definitely_not_registered")


def test_smolvla_explainer_is_registered():
    from lerobot.rollout.saliency import SmolVLAExplainer, get_explainer_cls

    assert get_explainer_cls("smolvla") is SmolVLAExplainer


def test_smolvla_explainer_produces_per_camera_maps():
    from lerobot.rollout.saliency import SmolVLAExplainer

    policy = _FakeSmolVLAPolicy()
    explainer = SmolVLAExplainer(policy)
    noise = torch.randn(1, 10, 4)
    chunk_kwargs = {"inference_delay": 2, "prev_chunk_left_over": None}

    maps = explainer.compute_saliency(_make_fake_batch(), noise, chunk_kwargs)

    assert set(maps) == {CAM_KEY}
    m = maps[CAM_KEY]
    assert m.shape == (32, 32)
    assert np.isfinite(m).all()
    assert (m >= 0).all()
    # chunk kwargs (incl. noise for exact replay) threaded through to the model
    assert policy.model.last_kwargs["inference_delay"] == 2
    assert policy.model.last_kwargs["noise"] is noise


def test_smolvla_explainer_map_depends_on_image():
    """Interior pixels of a uniform image yield a uniform map; a changed image changes it.

    The conv's zero padding makes the 1px border participate in fewer outputs, so
    uniformity is only asserted on the interior crop (评审 C2).
    """
    from lerobot.rollout.saliency import SmolVLAExplainer

    explainer = SmolVLAExplainer(_FakeSmolVLAPolicy())
    noise = torch.randn(1, 10, 4)
    kwargs = {"inference_delay": 0, "prev_chunk_left_over": None}

    uniform = _make_fake_batch()
    uniform[CAM_KEY] = torch.full((1, 3, 32, 32), 0.5)
    m0 = explainer.compute_saliency(uniform, noise, kwargs)[CAM_KEY]
    assert np.allclose(m0[2:-2, 2:-2], m0[2, 2])

    spotty = uniform.copy()
    spotty[CAM_KEY] = torch.rand(1, 3, 32, 32)
    m1 = explainer.compute_saliency(spotty, noise, kwargs)[CAM_KEY]
    assert not np.allclose(m1[2:-2, 2:-2], m1[2, 2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_saliency.py -v -k explainer`
Expected: FAIL — `ImportError: cannot import name 'SmolVLAExplainer'`

- [ ] **Step 3: Write minimal implementation**

创建 `src/lerobot/rollout/saliency/explainer.py`：

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Explainer registry: maps local policy types to saliency explainers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

_EXPLAINER_REGISTRY: dict[str, type] = {}


def register_explainer(policy_type: str, explainer_cls: type) -> type:
    """Register an Explainer class for a policy type (draccus ChoiceRegistry name)."""
    _EXPLAINER_REGISTRY[policy_type] = explainer_cls
    return explainer_cls


def get_explainer_cls(policy_type: str) -> type:
    """Look up the Explainer for a policy type. Fail fast — never silently degrade."""
    try:
        return _EXPLAINER_REGISTRY[policy_type]
    except KeyError:
        registered = ", ".join(sorted(_EXPLAINER_REGISTRY)) or "(none)"
        raise ValueError(
            f"No saliency explainer registered for policy type '{policy_type}'. "
            f"Registered types: {registered}. Implement and register an Explainer for this policy, "
            "or run without --saliency.enabled."
        ) from None


class Explainer:
    """Base class: wraps a local policy copy and computes per-camera saliency maps."""

    def __init__(self, policy) -> None:
        self.policy = policy

    def compute_saliency(
        self,
        batch: dict,
        noise: "torch.Tensor",
        chunk_kwargs: dict,
    ) -> dict[str, np.ndarray]:
        """Replay one chunk with gradients enabled.

        Args:
            batch: the preprocessed observation batch the policy acted on.
            noise: the exact noise tensor the executed chunk was sampled from.
            chunk_kwargs: the exact chunk kwargs (inference_delay, prev_chunk_left_over).

        Returns:
            {camera_key: (H, W) float32 saliency map at the batch image resolution}.
        """
        raise NotImplementedError
```

创建 `src/lerobot/rollout/saliency/smolvla.py`：

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SmolVLA Explainer: gradient replay of the executed action chunk (gradients × input).

Mirrors the call sequence of ``SmolVLAPolicy.predict_action_chunk`` → ``_get_chunk``
(src/lerobot/policies/smolvla/modeling_smolvla.py:225-241) but with the image inputs
marked ``requires_grad``. Passing the same ``noise`` + same batch + same chunk kwargs
reproduces the executed chunk deterministically (RTCProcessor holds no output-affecting state
(only an optional debug tracker), n_obs_steps=1; across GPU models FP kernels may differ at
ulp level — imperceptible for saliency),
so the gradient explains the action that was actually executed. Params stay frozen:
we differentiate wrt the input pixels only.
"""

from __future__ import annotations

import numpy as np
import torch

from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .explainer import Explainer, register_explainer


@register_explainer("smolvla")
class SmolVLAExplainer(Explainer):
    """Gradient×input saliency for SmolVLA."""

    def compute_saliency(
        self,
        batch: dict,
        noise: torch.Tensor,
        chunk_kwargs: dict,
    ) -> dict[str, np.ndarray]:
        policy = self.policy
        policy.eval()
        device = next(policy.parameters()).device

        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        noise = noise.to(device)
        chunk_kwargs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in (chunk_kwargs or {}).items()
        }

        batch = policy._prepare_batch(batch)
        policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])
        for k in batch:
            if k in policy._queues and k != ACTION:
                batch[k] = torch.stack(list(policy._queues[k]), dim=1)

        img_keys = [k for k in policy.config.image_features if k in batch]
        for k in img_keys:
            batch[k] = batch[k].detach().clone().requires_grad_(True)

        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **chunk_kwargs
        )
        actions = actions[:, :, : policy.config.action_feature.shape[0]]
        if policy.config.adapt_to_pi_aloha:
            actions = policy._pi_aloha_encode_actions(actions)

        # Backward target: squared L2 of the first executed action (batch is always 1 at
        # rollout). Same pixel map as the L2 norm up to a scalar that min-max removes.
        score = actions[0, 0].pow(2).sum()
        # allow_unused=True: a camera that didn't influence the action yields grad=None
        # instead of a RuntimeError (评审 I1 — with the default False it raises, never None).
        grads = torch.autograd.grad(score, [batch[k] for k in img_keys], allow_unused=True)

        saliency: dict[str, np.ndarray] = {}
        for key, grad in zip(img_keys, grads):
            if grad is None:  # camera had no influence on this action (e.g. padded mask)
                img = batch[key].detach()
                saliency[key] = np.zeros(img.shape[-2:], dtype=np.float32)
                continue
            img = batch[key].detach()
            if img.ndim == 5:
                img, grad = img[:, -1], grad[:, -1]
            cam_map = (grad * img).abs().sum(dim=1)[0]  # (C,H,W) → (H,W)
            saliency[key] = cam_map.float().cpu().numpy()
        return saliency
```

> **评审后补充（Task 3 评审，2026-09-10）**：实际提交的适配器在上述代码基础上追加了两处加固——
> 1. `policy = self.policy` 之后新增 `noise is None` 的显式契约错误（Task 5 快照可能产生 `noise=None`，Task 4 recorder 原样透传；需报可读错误而非 AttributeError）：
> ```python
>         if noise is None:
>             raise ValueError(
>                 "Saliency replay requires the injected noise tensor (snapshot came from an engine "
>                 "without noise injection); exact replay of the executed chunk is impossible"
>             )
> ```
> 2. 从 `policy._prepare_batch(batch)` 到 `torch.autograd.grad(...)` 的整段包进 `with torch.enable_grad():`（grad 模式是线程本地的，防止 recorder 线程泄漏的 `set_grad_enabled(False)` 逐帧炸出难排查的 RuntimeError）。另有一处文档措辞修正：module docstring 的 "RTCProcessor is stateless" 改为 "RTCProcessor holds no output-affecting state (only an optional debug tracker)"（上文片段已同步）。

更新 `src/lerobot/rollout/saliency/__init__.py` 为：

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Rollout-time saliency: per-camera input-pixel importance maps of the executed action."""

from .config import SaliencyConfig
from .explainer import Explainer, get_explainer_cls, register_explainer
from .smolvla import SmolVLAExplainer

__all__ = ["Explainer", "SaliencyConfig", "SmolVLAExplainer", "get_explainer_cls", "register_explainer"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_saliency.py -v`
Expected: 16 PASS（含 Task 1/2 的 12 个）

注意：`test_smolvla_explainer_produces_per_camera_maps` 断言 `policy.model.last_kwargs["noise"] is noise`——这要求适配器不做 noise clone/替换（只 `.to(device)`，CPU 输入时 `.to` 到同 device 返回原张量，`is` 成立；若断言失败，检查是否多了一次 clone）。

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/rollout/saliency/explainer.py src/lerobot/rollout/saliency/smolvla.py src/lerobot/rollout/saliency/__init__.py tests/test_saliency.py
git commit -m "feat(saliency): add Explainer registry and SmolVLA grad-replay adapter"
```

---

### Task 4: Rerun 推流 + SaliencyRecorder 旁路线程

**Files:**
- Modify: `src/lerobot/utils/rerun_visualization.py`（文件末尾追加函数）
- Create: `src/lerobot/rollout/saliency/recorder.py`
- Test: `tests/test_saliency.py`（追加）

- [ ] **Step 1: Write the failing test**

追加到 `tests/test_saliency.py`：

```python
# ---------------------------------------------------------------------------
# SaliencyRecorder (side thread, drop-oldest queue, per-camera mp4s)
# ---------------------------------------------------------------------------


class _FixedExplainer:
    def __init__(self):
        self.calls = 0

    def compute_saliency(self, batch, noise, chunk_kwargs):
        self.calls += 1
        h, w = batch[CAM_KEY].shape[-2:]
        return {CAM_KEY: np.full((h, w), 0.5, dtype=np.float32)}


def test_recorder_drops_stale_snapshots_and_writes_video(tmp_path):
    from lerobot.rollout.saliency.recorder import SaliencyRecorder

    explainer = _FixedExplainer()
    out = tmp_path / "saliency"
    recorder = SaliencyRecorder(
        explainer=explainer,
        output_dir=str(out),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=2.0,
        display=False,
        display_mode="rerun",
    )
    snap = {"preprocessed": _make_fake_batch(), "noise": torch.randn(1, 10, 4), "chunk_kwargs": {}}
    recorder.submit(snap)
    recorder.submit(snap)  # queue cap 1: first submission must be dropped
    recorder.start()
    recorder.close()

    assert explainer.calls == 1
    video = out / "cam1.mp4"  # filename = last path segment of the camera key
    assert video.exists() and video.stat().st_size > 0
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 1
    cap.release()


def test_recorder_close_is_idempotent(tmp_path):
    from lerobot.rollout.saliency.recorder import SaliencyRecorder

    recorder = SaliencyRecorder(
        explainer=_FixedExplainer(),
        output_dir=str(tmp_path),
        video_fps=10.0,
        alpha=0.5,
        blur_sigma=2.0,
        display=False,
        display_mode="rerun",
    )
    recorder.start()
    recorder.close()
    recorder.close()  # must not raise, must not double-release writers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_saliency.py -v -k recorder`
Expected: FAIL — `ModuleNotFoundError: No module named 'lerobot.rollout.saliency.recorder'`

- [ ] **Step 3: Write minimal implementation**

`src/lerobot/utils/rerun_visualization.py` 做四处修改（评审 I4：`log_rerun_data` 首次调用会发送自定义 blueprint，其中不包含 saliency 实体 → 不改这里，rerun viewer 默认**看不到**叠加帧）：

1. 模块级（`_ensure_blueprint` 定义之前）加路径注册表：

```python
# Saliency overlay entity paths seen so far; the blueprint is rebuilt (re-sent) whenever
# a new one arrives, so overlay views appear alongside the camera views without manual setup.
_SALIENCY_PATHS: set[str] = set()
```

2. `_build_blueprint`（line 76-92）把第一行 `views = [...]` 替换为：

```python
    views = [rrb.Spatial2DView(origin=path, name=path) for path in sorted(image_paths)]
    views.extend(
        rrb.Spatial2DView(origin=f"saliency/{cam}", name=f"saliency/{cam}")
        for cam in sorted(_SALIENCY_PATHS)
    )
```

3. `_ensure_blueprint`（line 95-108）在 `blueprint = _build_blueprint(...)` 之前加一行缓存路径集合（供 saliency 后到时重建 blueprint 用）：

```python
    log_rerun_data._last_paths = (observation_paths, action_paths, image_paths)
```

4. 文件末尾追加：

```python
def log_rerun_saliency(saliency_frames: dict[str, np.ndarray]) -> None:
    """Log saliency overlay frames (RGB uint8) under ``saliency/<camera>`` entities."""
    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    for cam, frame in saliency_frames.items():
        rr.log(f"saliency/{cam}", rr.Image(frame), static=True)

    new_paths = set(saliency_frames) - _SALIENCY_PATHS
    _SALIENCY_PATHS.update(saliency_frames)
    if new_paths and getattr(log_rerun_data, "blueprint", None) is not None:
        # The observation blueprint already ran (before the first saliency frame landed);
        # rebuild it from the stashed path sets so the new overlay views are included.
        obs_paths, act_paths, img_paths = getattr(log_rerun_data, "_last_paths", (set(), set(), set()))
        blueprint = _build_blueprint(obs_paths, act_paths, img_paths)
        log_rerun_data.blueprint = blueprint
        rr.send_blueprint(blueprint)
```

创建 `src/lerobot/rollout/saliency/recorder.py`：

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SaliencyRecorder: side thread that turns chunk snapshots into overlay frames, videos, and rerun entities.

Never blocks the control loop: the queue holds at most one snapshot (latest wins),
frames that arrive during a slow backward pass are dropped — the mp4s are diagnostic
tools, not measurement instruments (see CONTEXT.md: Saliency Overlay Video).
"""

from __future__ import annotations

import logging
import os
import queue
import threading

import cv2

from lerobot.utils.rerun_visualization import log_rerun_saliency

from .overlay import render_overlay, tensor_to_uint8_bgr

logger = logging.getLogger(__name__)


class SaliencyRecorder:
    """Side-thread pipeline: snapshot → saliency maps → overlay frame → mp4 + rerun."""

    def __init__(
        self,
        explainer,
        output_dir: str,
        video_fps: float,
        alpha: float,
        blur_sigma: float,
        display: bool,
        display_mode: str,
    ) -> None:
        self._explainer = explainer
        self._output_dir = output_dir
        self._video_fps = video_fps
        self._alpha = alpha
        self._blur_sigma = blur_sigma
        self._display = display
        self._display_mode = display_mode
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._writers: dict[str, object] = {}
        self._closed = False
        os.makedirs(output_dir, exist_ok=True)

    def submit(self, snapshot: dict | None) -> None:
        """Non-blocking, drop-oldest push. ``None`` is the shutdown sentinel."""
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(snapshot)
            except queue.Full:
                pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="saliency-recorder", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop the thread and flush the videos. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.submit(None)
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            if self._thread.is_alive():
                logger.warning("Saliency recorder thread did not exit within 30s; videos may be incomplete")

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                try:
                    self._process(item)
                except Exception:
                    logger.exception("Saliency frame dropped due to processing error")
        finally:
            # Writers are released here, on the recorder thread, so the main thread
            # never releases a writer the thread might still be writing to.
            for writer in self._writers.values():
                writer.release()

    def _process(self, item: dict) -> None:
        maps = self._explainer.compute_saliency(item["preprocessed"], item["noise"], item["chunk_kwargs"])
        rerun_frames: dict[str, object] = {}
        for cam_key, sal in maps.items():
            frame = tensor_to_uint8_bgr(item["preprocessed"][cam_key])
            overlay = render_overlay(frame, sal, self._alpha, self._blur_sigma)
            self._writer(cam_key, overlay.shape[1], overlay.shape[0]).write(overlay)
            if self._display and self._display_mode == "rerun":
                rerun_frames[cam_key.rsplit(".", 1)[-1]] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        if rerun_frames:
            log_rerun_saliency(rerun_frames)

    def _writer(self, cam_key: str, width: int, height: int):
        if cam_key in self._writers:
            return self._writers[cam_key]
        path = os.path.join(self._output_dir, f"{cam_key.rsplit('.', 1)[-1]}.mp4")
        for fourcc in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), self._video_fps, (width, height))
            if writer.isOpened():
                self._writers[cam_key] = writer
                return writer
            writer.release()
        raise RuntimeError(f"Could not open video writer for {path}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_saliency.py -v`
Expected: 18 PASS

注意：`_FixedExplainer` 的 `h, w = batch[CAM_KEY].shape[-2:]` 对 4 维 batch 张量 (B,C,H,W) 成立；`tensor_to_uint8_bgr` 已处理 4 维输入，两者分辨率一致，`render_overlay` 无需 resize。

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/utils/rerun_visualization.py src/lerobot/rollout/saliency/recorder.py tests/test_saliency.py
git commit -m "feat(saliency): add side-thread recorder writing per-camera overlay mp4s with rerun streaming"
```

> **评审后补充（Task 4 已实现后的修复与补测，见 9b884b2a 的评审）**：① 哨兵抢占竞争（close() 的 `None` 哨兵被并发 submit 从容量 1 的队列里 drop-oldest 抢走 → 线程永久阻塞 → join 30s 超时 → writers 永不释放，mp4 无 moov atom）：`_run` 改为 0.5s 轮询 + `queue.Empty` 时检查 `_closed` 退出，`submit` 加 `_closed` 顶卫（close 后丢弃真实快照、放行哨兵），回归测试 `test_recorder_survives_sentinel_steal_race`（先确认哨兵入队再触发竞争提交，已复现修复前的 15s 搁浅）；② rerun 变更补零覆盖：`tests/utils/test_rerun_visualization.py` 新增 `test_log_rerun_saliency_logs_entities`、`test_log_rerun_saliency_resends_blueprint_only_for_new_paths`；③ `init_rerun` 清空 `_SALIENCY_PATHS`（会话残留）；④ `_run` 连续两次 `_process` 失败升级为单条 error 日志并退出线程（成功重置计数）；⑤ `_writer` 增加 cam_key→path 冲突守卫（不同 cam_key 映射同一 mp4 路径 → RuntimeError）；⑥ `rerun_frames` 类型标注为 `dict[str, np.ndarray]`；⑦ `write()` 返回 `False` 时每相机告警一次（部分 OpenCV 构建返回 None，故用 `is False` 精确判断，避免每帧误报）。

---

### Task 5: RTC 引擎噪声注入 + 快照

**Files:**
- Modify: `src/lerobot/rollout/inference/rtc.py`（`__init__` 约 line 182-257 加属性；`_rtc_loop` line 503-505 改调用）
- Test: `tests/test_saliency.py`（追加）

- [ ] **Step 1: Write the failing test**

追加到 `tests/test_saliency.py`：

```python
# ---------------------------------------------------------------------------
# RTC engine: noise injection snapshot builder
# ---------------------------------------------------------------------------


def test_build_saliency_snapshot_clones_tensors():
    from lerobot.rollout.inference.rtc import _build_saliency_snapshot

    imgs = torch.rand(1, 3, 8, 8)
    noise = torch.rand(1, 10, 4)
    prev = torch.rand(4, 4)
    preprocessed = {CAM_KEY: imgs, "task": "pick"}

    snap = _build_saliency_snapshot(preprocessed, noise=noise, inference_delay=3, prev_chunk_left_over=prev)

    assert snap["preprocessed"][CAM_KEY] is not imgs
    assert snap["preprocessed"]["task"] == "pick"
    assert torch.equal(snap["noise"], noise)
    assert snap["chunk_kwargs"]["inference_delay"] == 3
    assert torch.equal(snap["chunk_kwargs"]["prev_chunk_left_over"], prev)
    snap["noise"].zero_()
    assert not torch.allclose(noise, torch.zeros_like(noise))  # deep copy, not a view


def test_build_saliency_snapshot_handles_none_optionals():
    from lerobot.rollout.inference.rtc import _build_saliency_snapshot

    snap = _build_saliency_snapshot({}, noise=None, inference_delay=0, prev_chunk_left_over=None)
    assert snap["noise"] is None
    assert snap["chunk_kwargs"]["prev_chunk_left_over"] is None
    assert snap["chunk_kwargs"]["inference_delay"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_saliency.py -v -k snapshot`
Expected: FAIL — `ImportError: cannot import name '_build_saliency_snapshot'`

- [ ] **Step 3: Write minimal implementation**

修改 `src/lerobot/rollout/inference/rtc.py`：

1. 模块 import 区加：
```python
from lerobot.policies.common.flow_matching import sample_noise
```

2. 模块级（`class RTCInferenceEngine` 定义之前）加快照构造器：
```python
def _build_saliency_snapshot(
    preprocessed: dict,
    noise: torch.Tensor | None,
    inference_delay: int,
    prev_chunk_left_over: torch.Tensor | None,
) -> dict:
    """Copy of everything the saliency explainer needs to replay one chunk exactly."""
    return {
        "preprocessed": {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in preprocessed.items()
        },
        "noise": None if noise is None else noise.detach().clone(),
        "chunk_kwargs": {
            "inference_delay": inference_delay,
            "prev_chunk_left_over": None
            if prev_chunk_left_over is None
            else prev_chunk_left_over.detach().clone(),
        },
    }
```

3. `RTCInferenceEngine.__init__`（在 `self._rtc_thread: Thread | None = None` 之后）加：
```python
        # Optional saliency sink (SaliencyManager). When set, chunks are generated with an
        # injected noise tensor (sampled here) and every consumed chunk is snapshotted to
        # the sink — the explainer replays it with the exact same noise for exact gradients.
        self.saliency_sink = None
```

4. `_rtc_loop` 中，把 line 503-505 的调用：
```python
                        actions = self._policy.predict_action_chunk(
                            preprocessed, inference_delay=delay, prev_chunk_left_over=prev_actions
                        )
```
替换为（**快照必须先于推理调用**——`_prepare_batch` → `_pi_aloha_decode_state` 会原地改写 `batch[OBS_STATE]`（`modeling_smolvla.py:384-385`），推理后快照会把已解码的 state 再解码一次，重放发散；评审 I2）：
```python
                        noise = None
                        if self.saliency_sink is not None:
                            noise = sample_noise(
                                (1, int(self._policy.config.chunk_size), int(self._policy.config.max_action_dim)),
                                policy_device,
                            )
                            if self._compile_warmup_done.is_set():
                                self.saliency_sink.submit(
                                    _build_saliency_snapshot(
                                        preprocessed,
                                        noise=noise,
                                        inference_delay=delay,
                                        prev_chunk_left_over=prev_actions,
                                    )
                                )

                        actions = self._policy.predict_action_chunk(
                            preprocessed, noise=noise, inference_delay=delay, prev_chunk_left_over=prev_actions
                        )
```

注意：
- `saliency_sink is None` 时 `noise` 为 `None`，`predict_action_chunk(noise=None)` 走原有内部采样路径——行为逐位不变。`policy_device` 在 `_rtc_loop` 开头已定义（line 407）。
- `is_warmup` 帧不提交（用 `self._compile_warmup_done`，compile 关闭时它恒为 set，不影响无 compile 路径）。
- 已知取舍（评审 M5）：快照发生在 epoch/discard 守卫（line 523-563）**之前**，被丢弃的 chunk（/reset 竞态、trained-RTC discard）也会产出热力图帧。base 策略无 reset 语义、丢弃极罕见，作为诊断工具可接受；不做修复。

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_saliency.py -v`
Expected: 20 PASS

再跑既有 RTC 测试确认无回归：
Run: `uv run pytest tests/policies/smolvla/test_smolvla_rtc.py tests/test_rollout.py -v`
Expected: 全部 PASS（与改动前相同）

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/rollout/inference/rtc.py tests/test_saliency.py
git commit -m "feat(saliency): inject reproducible noise in RTC engine and snapshot chunks to saliency sink"
```

**评审后补充**（Task 5 实施后的评审跟进，已随实施完成）：

1. **快照顺序注释**：提交块上方加了注释，说明快照必须在 `predict_action_chunk` 之前（`_prepare_batch` → `_pi_aloha_decode_state` 原地改写 `batch[OBS_STATE]`），防止未来"整理代码"把 submit 挪到调用之后而不出错地静默产出错误热力图。
2. **引擎级 sink 测试**（`tests/test_interactive_rollout.py`）：`test_rtc_engine_saliency_sink_noise_injection_and_snapshot_order` 走真实门控循环（`_StubChunkPolicy` 子类 + 录制 sink），断言 (a) 策略收到的 noise 与快照内 noise `torch.equal`、(b) 首个 submit 事件先于首个 call 事件、(c) 快照三键 + 张量深拷贝（非张量按引用）、(d) noise 形状 `(1, chunk_size, max_action_dim)` float32。控制路径由此获得首个自动化覆盖。
3. **sink 故障隔离**（rtc.py）：`sample_noise`/`submit` 块包 try/except——失败时 log 一条 warning（`exc_info=True`）并永久置 `self.saliency_sink = None`、`noise = None`，降级到禁用路径；坏掉的旁路线程不再累积 consecutive-error 击落 RTC 线程。配套测试 `test_rtc_engine_sink_failure_disables_saliency`。
4. **测试保真**：`test_build_saliency_snapshot_clones_tensors` 的 `"task"` 改为列表（`["pick"]`，贴近真实 preprocessed 形状），并加断言 `snap["preprocessed"]["task"] is preprocessed["task"]` 记录非张量按引用契约。

实施时的一处必要偏离（偏离评审原文、经核实属必需）：调用点为 `**({"noise": noise} if noise is not None else {})` 而非无条件 `noise=noise`——`supports_rtc_inference` 探针只绑定 `inference_delay`/`prev_chunk_left_over`，存根策略的 `predict_action_chunk` 无 `noise` 形参；无条件传 `noise=None` 会让这类策略 TypeError（`tests/test_interactive_rollout.py` 两个既有测试因此失败，已用 stash 基线验证）。禁用路径下省略该 kwarg 才与今日行为逐位一致。

---

### Task 6: SaliencyManager + context/rollout script 接线

**Files:**
- Create: `src/lerobot/rollout/saliency/manager.py`
- Modify: `src/lerobot/rollout/saliency/__init__.py`
- Modify: `src/lerobot/rollout/context.py`（`PolicyContext` line 223-230 加字段；`build_rollout_context` 在 line 345（"Policy loaded" 日志）后加 manager 创建——必须在 line 338-341 的 RTC 注入之后，见 Step 3 的 C3 说明；引擎创建后约 line 598 挂 sink + 传入 PolicyContext）
- Modify: `src/lerobot/scripts/lerobot_rollout.py`（`finally` 块约 line 259-262）
- Test: `tests/test_saliency.py`（追加）

- [ ] **Step 1: Write the failing test**

追加到 `tests/test_saliency.py`：

```python
# ---------------------------------------------------------------------------
# SaliencyManager: device resolution + registry fail-fast
# ---------------------------------------------------------------------------


def test_resolve_explainer_device_prefers_second_gpu(monkeypatch):
    from lerobot.rollout.saliency.manager import resolve_explainer_device

    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    assert resolve_explainer_device(None, "cuda:0") == "cuda:1"
    assert resolve_explainer_device(None, "cpu") == "cuda:0"
    assert resolve_explainer_device("cuda:1", "cuda:0") == "cuda:1"  # explicit wins

    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    assert resolve_explainer_device(None, "cuda:0") == "cuda:0"  # fallback: same device
    assert resolve_explainer_device(None, "cpu") == "cpu"


def test_saliency_manager_fails_fast_on_unregistered_policy(tmp_path):
    from lerobot.rollout.saliency.manager import SaliencyManager

    def _load():
        raise AssertionError("loader must not run when the policy type is unregistered")

    cfg = SimpleNamespace(device=None, output_dir=str(tmp_path), video_fps=10.0, alpha=0.5, blur_sigma=2.0, display=False)
    policy_config = SimpleNamespace(type="definitely_not_registered")

    with pytest.raises(ValueError, match="No saliency explainer registered"):
        SaliencyManager(cfg, policy_config, policy_device="cpu", display_mode="rerun", load_policy=_load)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_saliency.py -v -k "device or fails_fast_on_unregistered"`
Expected: FAIL — `ModuleNotFoundError: No module named 'lerobot.rollout.saliency.manager'`

- [ ] **Step 3: Write minimal implementation**

创建 `src/lerobot/rollout/saliency/manager.py`：

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SaliencyManager: owns the explainer's model copy on a dedicated GPU and the recorder.

The control loop stays on its own GPU under no_grad; this model copy lives on another
device (when one exists) and replays chunks with gradients enabled. See CONTEXT.md:
Explainer.
"""

from __future__ import annotations

import logging
import os
import time

import torch

from .explainer import get_explainer_cls
from .recorder import SaliencyRecorder

logger = logging.getLogger(__name__)


def resolve_explainer_device(cfg_device: str | None, policy_device: str) -> str:
    """Explicit --saliency.device wins; else a second GPU if present; else the policy device."""
    if cfg_device:
        return cfg_device
    if torch.cuda.device_count() > 1:
        if policy_device in ("cuda", "cuda:0"):
            return "cuda:1"
        # Policy on CPU (or cuda:1) with spare GPUs: give the explainer cuda:0 — any GPU
        # beats sharing the policy's device; explicit --saliency.device overrides this.
        return "cuda:0"
    return policy_device


class SaliencyManager:
    """Lifecycle owner for the saliency sidecar: model copy, explainer, recorder."""

    def __init__(
        self,
        saliency_cfg,
        policy_config,
        policy_device: str,
        display_mode: str,
        load_policy,
    ) -> None:
        explainer_cls = get_explainer_cls(policy_config.type)  # fail fast, before weights load

        self.device = resolve_explainer_device(saliency_cfg.device, policy_device)
        policy_copy = load_policy()
        policy_copy.to(self.device)
        policy_copy.eval()
        for p in policy_copy.parameters():
            p.requires_grad_(False)  # differentiate wrt input pixels only
        self._explainer = explainer_cls(policy_copy)

        output_dir = saliency_cfg.output_dir or os.path.join(
            "outputs", "saliency", time.strftime("%Y%m%d_%H%M%S")
        )
        logger.info(
            "Saliency explainer ready: policy copy on %s, cameras %s, output %s",
            self.device,
            list(getattr(policy_config, "input_features", {})),
            output_dir,
        )
        self._recorder = SaliencyRecorder(
            explainer=self._explainer,
            output_dir=output_dir,
            video_fps=saliency_cfg.video_fps,
            alpha=saliency_cfg.alpha,
            blur_sigma=saliency_cfg.blur_sigma,
            display=saliency_cfg.display,
            display_mode=display_mode,
        )
        self._recorder.start()

    def submit(self, snapshot: dict | None) -> None:
        """Sink entry point called by the RTC engine (drop-oldest, never blocks)."""
        self._recorder.submit(snapshot)

    def stop(self) -> None:
        """Flush and close the overlay videos."""
        self._recorder.close()
```

`__init__.py` **不再改动**（评审 M2：`configs.py` 顶部 import 会执行包 `__init__`，若 `__init__` 再引 manager→recorder→cv2/rerun，默认路径就背上整个 sidecar——保持 __init__ 只含 config/explainer/smolvla 这些轻量导出，manager 一律经子模块 `lerobot.rollout.saliency.manager` 引用）。

修改 `src/lerobot/rollout/context.py`：

1. `PolicyContext`（line 223-230）加字段（dataclass 字段区末尾）：
```python
    saliency: SaliencyManager | None = None
```
并在文件顶部 TYPE_CHECKING import 区加：
```python
    from .saliency.manager import SaliencyManager
```

2. `build_rollout_context` 中，`logger.info("Policy loaded: type=%s, device=%s", ...)`（line 345）之后、`torch_compile_active` 段之前插入。

> **为什么必须在这里（评审 C3，load-bearing）**：运行时 RTC 配置注入（`policy.config.rtc_config = cfg.inference.rtc` + `init_rtc_processor()`）发生在 line 338-341，且 `from_pretrained` **共享 config 对象**（`policies/pretrained.py:97`）。lerobot-train 保存的 checkpoint `rtc_config=None`——若在注入前加载副本，副本 `rtc_processor=None` 但共享 config 后来翻转为 enabled，重放每帧 `AttributeError`，一个热力图都出不来。插在 line 345 之后保证副本加载时 config 已带 RTC；此时硬件尚未初始化，fail-fast 语义不变。
```python
    saliency_manager = None
    if cfg.saliency.enabled:
        # Submodule import (not the package __init__): keeps cv2/rerun out of the default import path.
        from lerobot.rollout.saliency.manager import SaliencyManager

        def _load_explainer_policy():
            # torch.compile must stay off the explainer's copy.
            prev = getattr(policy_config, "compile_model", None)
            if prev is not None:
                policy_config.compile_model = False
            try:
                return _load_pretrained_policy(policy_config)
            finally:
                if prev is not None:
                    policy_config.compile_model = prev

        logger.info("Saliency enabled — loading explainer model copy...")
        saliency_manager = SaliencyManager(
            cfg.saliency,
            policy_config,
            policy_device=cfg.device or "cpu",
            display_mode=cfg.display_mode,
            load_policy=_load_explainer_policy,
        )
```

3. 引擎创建之后（`inference_strategy = create_inference_engine(...)` 与 `# --- 8. Assemble` 之间）插入：
```python
    if saliency_manager is not None:
        # Validation guarantees cfg.inference is RTC when saliency is enabled.
        inference_strategy.saliency_sink = saliency_manager
```

4. `RolloutContext(...)` 组装处（line 598 的 `PolicyContext(...)`）加一个关键字参数：
```python
        policy=PolicyContext(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            inference=inference_strategy,
            saliency=saliency_manager,
        ),
```

修改 `src/lerobot/scripts/lerobot_rollout.py` 的 `finally` 块（line 262-265）：

```python
    finally:
        strategy.teardown(ctx)
        if ctx.policy.saliency is not None:
            logger.info("Finalizing saliency overlay videos...")
            ctx.policy.saliency.stop()
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_saliency.py -v`
Expected: 22 PASS

全量回归（rollout + 引擎相关）：
Run: `uv run pytest tests/test_rollout.py tests/test_interactive_rollout.py tests/policies/smolvla/ -v`
Expected: 与改动前一致，无新失败

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/rollout/saliency/manager.py src/lerobot/rollout/saliency/__init__.py src/lerobot/rollout/context.py src/lerobot/scripts/lerobot_rollout.py tests/test_saliency.py
git commit -m "feat(saliency): add SaliencyManager and wire it into rollout context lifecycle"
```

#### 评审后补充（Task 6 质量评审，commit 2c48d953）

- **Important 1 — 消除 explainer 副本加载时的策略 GPU 驻留尖峰**：`from_pretrained` 按 `config.device` 加载权重（policies/pretrained.py:183/197/203），而该 config 是共享对象——原来副本先落在策略 GPU 上再 `.to(cuda:1)`，启动瞬时驻留 ~2× 模型大小，恰在双卡场景 OOM。修复：`context.py` 里先 `resolve_explainer_device` 得到目标设备，闭包内在 save/restore 包装中额外覆盖 `policy_config.device`，让 `from_pretrained` 直接加载到目标 GPU；同时把解析结果写回 `cfg.saliency.device`（显式设备优先，幂等），manager 与闭包按构造一致。
- **Minor 2 — `resolve_explainer_device` 兜底分支收窄**：原注释与行为不符（任何非 cuda:0 设备都会被派到 cuda:0）。现仅 `cpu`/`cuda:1` 兜底到另一张卡；`mps`/`xpu` 等保持原设备（跨厂商默认太意外）。测试补 `mps` 断言。
- **Minor 3 — 类型注解**：`SaliencyManager.__init__` 的 `saliency_cfg: SaliencyConfig`（从 `.config` 导入，无重依赖）与 `load_policy: Callable[[], torch.nn.Module]`；SimpleNamespace 测试不受影响（运行时鸭子类型）。
- **Minor 4 — 构建中途失败泄漏 recorder 线程**：`build_rollout_context` 在 manager 创建之后的全部剩余逻辑包进 `try`，`except Exception` 里 `saliency_manager.stop()` 后 `raise`——机器人连接/引擎创建失败不再留下半开的 mp4 写手与存活线程。

---

### Task 7: rollout.sh 示例 + 全量检查

**Files:**
- Modify: `rollout.sh`

- [ ] **Step 1: 在 rollout.sh 末尾的 lerobot-rollout 调用参数里追加注释掉的可选开关**（不改变现有默认行为）。

> **评审 I3**：`rollout.sh:5` 现有 `export CUDA_VISIBLE_DEVICES=2`——进程内只见一张卡，"explainer 自动挑第二张 GPU"不会触发。启用 saliency 时必须同时把它改成多卡掩码（如 `export CUDA_VISIBLE_DEVICES=2,3`），或显式 `--saliency.device=cuda:1`。

```bash
  # --- Optional: saliency overlay videos (per-camera mp4 + rerun live view) ---
  # Requires a second GPU for the explainer model copy; falls back to the policy GPU otherwise.
  # Videos land in outputs/saliency/<timestamp>/{camera1,camera2,camera3}.mp4
  # --saliency.enabled=true \
  # --display_data=true --display_mode=rerun \
```

（追加到现有参数列表尾部、反斜杠续行之前——保持注释行不参与续行链：若现有最后一行参数以 `\` 结尾，注释行放在其后即可，因为被注释的行不续行。）

- [ ] **Step 2: 全量测试**

Run: `uv run pytest tests/test_saliency.py tests/test_rollout.py -v`
Expected: 全部 PASS

- [ ] **Step 3: pre-commit**

Run: `pre-commit run --files $(git diff --name-only HEAD~6) 2>&1 | tail -20`（或 `pre-commit run --all-files`）
Expected: 无新增 failure（ruff 格式如有自动修复，确认后重新 `git add`）

- [ ] **Step 4: Commit**

```bash
git add rollout.sh
git commit -m "docs(saliency): add commented saliency flags to rollout.sh"
```

---

### Task 8: 上机 E2E 验证（需要用户参与，不能自动化）

这是唯一验证“真 SmolVLA 权重 + 真机器人”路径的环节（repo 内无可 CI 化的完整 SmolVLA batch 先例）。

- [ ] **Step 0: 调整 GPU 掩码（评审 I3）**：`rollout.sh:5` 的 `export CUDA_VISIBLE_DEVICES=2` 让进程只见一张卡——改为 `export CUDA_VISIBLE_DEVICES=2,3`（或按机器实际卡号），否则 explainer 永远与控制线程共卡，"第二 GPU"验收项全部作废。

- [ ] **Step 1: 确认硬件**：`nvidia-smi` 应显示物理机 ≥2 张可用卡，且 `CUDA_VISIBLE_DEVICES` 掩码包含其中两张；确认 `models/train/smolvla/.../pretrained_model` 路径存在。下文中的 cuda:0/cuda:1 指**进程可见**的编号，不是物理卡号。

- [ ] **Step 2: 在 rollout.sh 中取消注释 saliency 两行，跑一次短 rollout**（`duration=60` 左右），同时开 rerun viewer：
```bash
bash rollout.sh
```
Expected:
- 启动日志出现 `Saliency explainer ready: policy copy on cuda:1 ...`（cuda:1 = 掩码内的第二张可见卡）
- rerun viewer 中 `saliency/camera1|camera2|camera3` 实体出现叠加帧
- 控制循环 cadence 报告与不开 saliency 时一致（explainer 不阻塞控制循环）
- E2E 必须覆盖 `prev_chunk_left_over != None` 的帧（steady-state rollout 稳定运行数分钟后即满足——首个 chunk 之后所有 chunk 都带 leftover 前缀）

- [ ] **Step 3: Ctrl-C 结束后检查产物**：
```bash
ls -la outputs/saliency/<timestamp>/
```
Expected: 3 个 mp4，每个 ≥1 帧、能正常播放、turbo 热区叠加在原始画面上。

- [ ] **Step 4: 显存与延迟体检**：rollout 中 `nvidia-smi` 确认掩码内第二张卡（进程视角 cuda:1）多出 ~2GB 级占用，且策略那张卡（进程视角 cuda:0）占用不变；如策略卡显存异常上涨，说明 explainer 误用了策略 GPU（检查 `--saliency.device`）。

- [ ] **Step 5: 全部通过后按需把开关合入 rollout.sh 默认参数并提交**（用户决定是否默认开启）。

---

## Self-Review 记录

### 第一轮：写计划后自查（见下方原记录）

- **Spec coverage**: 共识清单逐项对应——gradients×input（Task 3）、聚合到执行动作第一帧 + L2（Task 3 score 目标）、严格 noise 注入复现（Task 5）、旁路线程不阻塞 + 队列容量 1（Task 4）、每相机 mp4 x264 + 720p + turbo + alpha 0.5 + min-max + σ2 模糊（Task 2/4）、rerun 实体（Task 4）、Ctrl-C 优雅 flush（Task 4 close）、第二 GPU 模型副本（Task 6）、未注册策略 fail-fast（Task 1 config 校验 + Task 3/6 注册表）、`--saliency.*` 配置（Task 1）、分支（已建 `feat/rollout-saliency`）。DM05/openpi 排除（无任务，符合共识）。
- **已知偏差（须在实现中注意）**: ① x264 编码在 OpenCV 上依赖本地构建，计划用 `avc1`→`mp4v` 回退链兜底（Task 4 `_writer`）；② 真 SmolVLA 的适配器保真度由 FakePolicy 单测 + Task 8 E2E 共同覆盖；③ "当前执行的那一步" 实现为 chunk 首个 action（`actions[0, 0]`），RTC execution_horizon 下它就是下一个被消耗的动作，符合共识 Q2。
- **Type consistency**: `compute_saliency(batch, noise, chunk_kwargs)` 签名在 explainer.py / smolvla.py / recorder._process / 测试四处一致；`submit(snapshot)` / `close()` 在 manager 与 recorder 间一致；`_build_saliency_snapshot` 的键（`preprocessed` / `noise` / `chunk_kwargs`）与 recorder 消费端一致。

### 第二轮：code-reviewer 评审后的修复记录（评审发现 3C + 5I + 7M）

全部 Critical/Important 已修复进上方各任务，Minor 中除以下两项外也已处理：
- **C1/C2**（Task 3）：FakeSmolVLAPolicy 改为 `torch.nn.Module` 子类；均匀图断言改为内部裁剪 `m[2:-2, 2:-2]`（conv 零填充使 1px 边框梯度天然不均匀）。
- **C3**（Task 6）：manager 创建点移到 `context.py:345`（RTC 配置注入之后），并在计划中写明共享 config 对象这一事实——这是三个 Critical 里唯一一个"不改就整条功能静默失效"的缺陷。
- **I1**（Task 3）：`torch.autograd.grad(..., allow_unused=True)`，None 检查移到切片之前。
- **I2**（Task 5）：快照提交移到 `predict_action_chunk` 调用之前（`_pi_aloha_decode_state` 原地改写 state）。
- **I3**（Task 7/8）：`CUDA_VISIBLE_DEVICES=2` 单卡掩码问题，Task 8 新增 Step 0。
- **I4**（Task 4）：rerun blueprint 重发机制（`_SALIENCY_PATHS` + 路径缓存 + 新路径到达时重建 blueprint）。
- **I5**（Task 1）：测试文件 import 全部置顶的纪律写进 Task 1。
- **M1-M3/M5-M7**：行号锚点收紧、`__init__` 保持轻量（manager 走子模块 import）、torch.compile 警告、丢弃 chunk 取舍说明、display_data 校验、cpu 回落注释。
- **未修复**：M4 的措辞已软化（"确定性重放"替代"逐位复现"），但跨 GPU 型号的 ulp 级差异客观存在，无法根除——对热力图无感知影响，接受。测试计数相应更新为 7/12/16/18/20/22。

### 第三轮：终审（a6c6cdde..57cc03b1 全分支）后的修复记录（0C + 0I + 5M，另采纳 2 条建议）

结论 **approve-with-minor**。终审额外做了实证：真实（随机初始化、2 层 VLM）SmolVLA 在 CPU 上驱动 explainer 重放，动作与 `predict_action_chunk` 逐位一致（max diff 0.0），含 RTC guidance 路径。修复全部落在 `ae59ae82`：

- **M2 → 修复**：新增 `accepts_noise_injection` 探测（rtc.py），context.py 挂 sink 前构建期检查——RTC 兼容但无 noise 参数的策略未来注册 explainer 时在构建期报错，不再拖到控制循环里 TypeError。
- **M3 → 修复**：`_build_blueprint` 遍历 `sorted(_SALIENCY_PATHS.copy())`（GIL 下原子快照，消除主线程遍历 vs recorder 线程写入的理论竞态）。
- **M4 → 修复**：`tensor_to_uint8_bgr` 支持 5 维 `(B,T,C,H,W)`（取最新一帧），消除与未来带观测队列策略的不对称。
- **M1 → 决策**：CONTEXT.md 按 grill-with-docs / domain-modeling 工作流提交到仓库根目录（该工作流规定 CONTEXT.md 是仓库根部的领域词汇表，本应入库）。
- **M5 → 修复**：三处注释/日志漂移（manager 只把 VISUAL 特征记为 cameras；rtc.py 注释改为"每个**产出**的 chunk 被快照"；context.py import 注释改为如实描述条件导入）。
- **建议 1 → 采纳**：recorder 对非有限热力值丢帧 + 每相机 warn-once（防 FP 异常产出看似正常的垃圾视频）。
- **建议 2 → 采纳**：终审的实证脚本固化为 `test_smolvla_explainer_replays_real_policy_bitwise`（transformers importorskip 守护，CPU 数秒），风险最高的接缝从此有 CI 保护。

终审后测试：88 passed（test_saliency 31 + rerun 可视化 + interactive_rollout），ruff/format/bandit/typos 全净。

### 第四轮：上机 E2E 后的用户反馈修订（2026-09-11）

- **E2E 通过**：完整 rollout 成功，3 个 mp4 生成（avc1 硬编不可用，按设计回退 mp4v），rerun 实时叠加正常，控制回路 29.91Hz / 0 超预算 tick。
- **归一化尺度修订**：共识第 3 轮定的"逐帧 min-max"原本是**每相机独立**归一化——用户上机后指出这会把"几乎没贡献的相机"内部显著像素染成深红，三个视角颜色不可比。修订为：**同一 chunk 的所有相机共用一套 (min, max)**（`shared_norm_range`，blur 后取跨相机极值），贡献小的视角保持深色。rerun 实时画面与 mp4 走同一路径，自动一致。时间维度仍逐 chunk 独立（未采用全局运行尺度——早期尖峰会压缩后续全部动态范围）。
- 附带修复：`shared_norm_range` 初版对 generator 迭代两次（min 耗尽后 max 抛 ValueError），单测已固化 generator 输入。
- **时间维度尺度修订**：跨相机共享后用户追问时间轴——原实现逐 chunk 独立 min-max，"平静时刻"（整体梯度小）会被拉满量程染红，无法区分"策略真的高度依赖视觉"与"不敏感但被放大"。修订为 `--saliency.norm_scope`：默认 `run`（lo 恒 0——热力值非负；hi = 每 chunk 衰减 0.99 的运行最大值，尖峰约 1 分钟恢复，颜色深浅≈绝对贡献，跨相机跨时刻可比），`chunk` 恢复逐 chunk 自适应。已知取舍：mp4 流式写入无法回头重染，run 尺度在线收敛、单向缓慢漂移；完美的两遍全局尺度不可行（rollout 不录原始帧）。
- **梯度目标语义确认（不改代码）**：用户质疑 `‖a₀‖²` 目标让热力图受舵机角度值影响。讨论后用户明确"想要的就是哪些像素最能放大/缩小当前指令的幅度"——这正是 ∂‖a₀‖²/∂x = 2Σ aⱼ(∂aⱼ/∂x) 的语义，动作值加权是特性而非缺陷。备选（动作和 Σaⱼ / 随机探针 / 精确逐维 12×）已记录但否决。已接受推论：归一化动作过零位（≈数据集均值姿态）时全图变暗 = "此刻没有可被视觉放大的指令幅度"，在 run-scope 尺度下如实呈现。
- **逐维取 max 聚合（否决）**：用户提出逐像素取各动作维度贡献的绝对值最大（maxⱼ|aⱼ·gⱼ·x|），可修跨关节符号抵消的盲点（A 关节 +g、B 关节 −g 的像素在求和下消失），但逐维梯度无法从单次反传分离（一次 backward 只给出 Jacobian 行的线性组合），需每维一次反传（12×；top-3 按 |aⱼ| 近似 3×；vmap 批量约 3~6× wall time）。用户决定保持求和，不付此开销/帧率代价。若未来重议：注意 max 不奖励"同时影响多个关节"的集体效应像素——sum 漏方向相反者，max 漏广泛使用者。
