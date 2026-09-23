"""Compare native PI paths with the immutable pre-saliency revision.

Only dimensions are reduced: real vision, Gemma, cache, flow matching and backward
code run on both sides. No downloads or pretrained weights are needed. GPU compile
checks are opt-in: SALIENCY_COMPILE_TEST=1 pytest ... -k compile.
"""

import ast
import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

BASELINE = "9706a20c0116ff99f48d4d24af98dc50704df786"
ROOT = Path(__file__).resolve().parents[2]


def baseline_module(name):
    path = "src/" + name.replace(".", "/") + ".py"
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT, text=True)
    module = types.ModuleType(name + "_saliency_baseline")
    module.__package__ = name.rsplit(".", 1)[0]
    sys.modules[module.__name__] = module
    exec(compile(source, f"{BASELINE}:{path}", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("filename", ["modeling_smolvla.py", "smolvlm_with_expert.py"])
def test_smolvla_training_bodies_unchanged(filename):
    """Smol changes are inference bookkeeping; training code remains bytecode-equivalent."""
    path = "src/lerobot/policies/smolvla/" + filename
    before = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT, text=True)
    after = (ROOT / path).read_text()

    # Compare syntax, ignoring comments/locations, for every training/attention method.
    def training_bodies(source):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
            and node.name
            in {
                "forward",
                "train",
                "set_requires_grad",
                "forward_attn_layer",
                "forward_cross_attn_layer",
                "eager_attention_forward",
            }
        }

    assert training_bodies(before) == training_bodies(after)


def miniature(module, monkeypatch):
    """Shrink dimensions before construction without replacing model operations."""
    monkeypatch.setattr(module, "get_gemma_config", lambda _: module.GemmaConfig(32, 2, 64, 8, 1, 4))
    original_vlm = module.PaliGemmaForConditionalGenerationWithPiGemma
    original_expert = module.PiGemmaForCausalLM

    def vlm(config):
        config.text_config.vocab_size = 64
        config.image_token_index = 63
        config.projection_dim = 32
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_hidden_layers = 2
        config.vision_config.num_attention_heads = 4
        config.vision_config.projection_dim = 32
        return original_vlm(config)

    def expert(config):
        config.vocab_size = 64
        return original_expert(config)

    monkeypatch.setattr(module, "PaliGemmaForConditionalGenerationWithPiGemma", vlm)
    monkeypatch.setattr(module, "PiGemmaForCausalLM", expert)


def models(kind, monkeypatch, *, freeze="none", checkpoint=False, device="cpu"):
    if kind == "smolvla":
        return smol_models(monkeypatch, freeze=freeze, device=device)
    name = f"lerobot.policies.{kind}.modeling_{kind}"
    current = importlib.import_module(name)
    old_gemma = baseline_module("lerobot.policies.pi_gemma")
    old = baseline_module(name)
    old.PaliGemmaForConditionalGenerationWithPiGemma = old_gemma.PaliGemmaForConditionalGenerationWithPiGemma
    old.PiGemmaForCausalLM = old_gemma.PiGemmaForCausalLM
    old.euler_integrate = baseline_module("lerobot.policies.common.flow_matching").euler_integrate
    config_cls = getattr(current, kind.upper() + "Config")
    config = config_cls(
        device=device,
        dtype="float32",
        image_resolution=(28, 28),
        chunk_size=4,
        n_action_steps=4,
        max_action_dim=4,
        max_state_dim=4,
        num_inference_steps=2,
        freeze_vision_encoder=freeze == "vision",
        train_expert_only=freeze == "expert",
    )
    result = []
    for module in (old, current):
        miniature(module, monkeypatch)
        torch.manual_seed(419)
        model = getattr(module, kind.upper() + "Pytorch")(config).to(device)
        if checkpoint:
            model.gradient_checkpointing_enable()
        result.append(model)
    return result


def smol_models(monkeypatch, *, freeze, device):
    from transformers import SmolVLMConfig

    from lerobot.policies.smolvla import modeling_smolvla as current, smolvlm_with_expert as current_vlm
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    old = baseline_module("lerobot.policies.smolvla.modeling_smolvla")
    old_vlm = baseline_module("lerobot.policies.smolvla.smolvlm_with_expert")
    old.SmolVLMWithExpertModel = old_vlm.SmolVLMWithExpertModel
    old.euler_integrate = baseline_module("lerobot.policies.common.flow_matching").euler_integrate
    result = []
    for module, vlm_module in ((old, old_vlm), (current, current_vlm)):
        config = SmolVLMConfig(
            text_config={
                "model_type": "llama",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "vocab_size": 64,
                "pad_token_id": 0,
            },
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 28,
                "patch_size": 14,
            },
            image_token_id=63,
            scale_factor=2,
        )
        monkeypatch.setattr(
            vlm_module, "AutoConfig", types.SimpleNamespace(from_pretrained=lambda _, c=config: c)
        )
        processor = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(fake_image_token_id=61, global_image_token_id=62)
        )
        monkeypatch.setattr(
            vlm_module, "AutoProcessor", types.SimpleNamespace(from_pretrained=lambda _, p=processor: p)
        )
        cfg = SmolVLAConfig(
            device=device,
            chunk_size=4,
            n_action_steps=4,
            max_action_dim=4,
            max_state_dim=4,
            num_steps=2,
            num_vlm_layers=2,
            num_expert_layers=2,
            expert_width_multiplier=1.0,
            load_vlm_weights=False,
            freeze_vision_encoder=freeze == "vision",
            train_expert_only=freeze == "expert",
        )
        torch.manual_seed(419)
        result.append(module.VLAFlowMatching(cfg).to(device))
    return result


def inputs(kind, device):
    torch.manual_seed(53)
    kw = {
        "images": [torch.rand(1, 3, 28, 28, device=device)],
        "img_masks": [torch.ones(1, dtype=torch.bool, device=device)],
        "noise": torch.randn(1, 4, 4, device=device),
    }
    tokens = torch.randint(0, 64, (1, 3), device=device)
    masks = torch.ones_like(tokens, dtype=torch.bool)
    if kind in ("pi0", "smolvla"):
        kw.update(lang_tokens=tokens, lang_masks=masks, state=torch.randn(1, 4, device=device))
    else:
        kw.update(tokens=tokens, masks=masks)
    return kw


@pytest.mark.parametrize(
    "kind,checkpoint", [("pi0", False), ("pi0", True), ("pi05", False), ("pi05", True), ("smolvla", False)]
)
@pytest.mark.parametrize("freeze", ["none", "vision", "expert"])
def test_training_matches_pre_saliency_revision(kind, freeze, checkpoint, monkeypatch):
    old, new = models(kind, monkeypatch, freeze=freeze, checkpoint=checkpoint)
    assert old.state_dict().keys() == new.state_dict().keys()
    assert {n: p.requires_grad for n, p in old.named_parameters()} == {
        n: p.requires_grad for n, p in new.named_parameters()
    }
    kw = inputs(kind, "cpu")
    kw.update(actions=torch.randn(1, 4, 4), time=torch.tensor([0.4]))
    results = []
    for model in (old, new):
        model.train()
        loss = model(**kw).mean()
        loss.backward()
        results.append(
            (
                loss.detach(),
                {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()},
            )
        )
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name, gradient in results[0][1].items():
        actual = results[1][1][name]
        if gradient is None:
            assert actual is None, name
        else:
            torch.testing.assert_close(actual, gradient, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("kind", ["pi0", "pi05", "smolvla"])
def test_default_sampling_matches_pre_saliency_revision(kind, monkeypatch):
    old, new = models(kind, monkeypatch)
    kw = inputs(kind, "cpu")
    old.eval()
    new.eval()
    before = torch.random.get_rng_state()
    with torch.no_grad():
        expected = old.sample_actions(**kw)
        actual = new.sample_actions(**kw)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad
    capture_owner = new.vlm_with_expert if kind == "smolvla" else new
    assert capture_owner.attention_capture is None


@pytest.mark.parametrize("guided", [False, True])
def test_shared_rtc_default_matches_baseline(guided):
    from lerobot.policies.common import flow_matching
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

    old_flow = baseline_module("lerobot.policies.common.flow_matching")
    old_rtc = baseline_module("lerobot.policies.rtc.modeling_rtc")
    torch.manual_seed(24)
    noise = torch.randn(1, 6, 2)
    leftover = torch.randn(1, 4, 2) if guided else None
    outputs = []
    for integrate, processor_cls in (
        (old_flow.euler_integrate, old_rtc.RTCProcessor),
        (flow_matching.euler_integrate, RTCProcessor),
    ):
        parameter = torch.tensor(0.3, requires_grad=True)
        result = integrate(
            lambda x, t, p=parameter: p * x.sin(),
            noise,
            3,
            rtc_processor=processor_cls(RTCConfig(execution_horizon=4)),
            rtc_enabled=True,
            inference_delay=1,
            prev_chunk_left_over=leftover,
        )
        gradient = torch.autograd.grad(result.sum(), parameter)[0]
        outputs.append((result.detach(), gradient))
    for expected, actual in zip(*outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_queue_cropping_and_provenance_match_baseline(enabled):
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    old_cls = baseline_module("lerobot.policies.rtc.action_queue").ActionQueue
    queues = [cls(RTCConfig(enabled=enabled)) for cls in (old_cls, ActionQueue)]
    for delay in (0, 2, 8):
        actions = torch.arange(12).reshape(6, 2).float()
        for queue in queues:
            queue.merge(actions, actions + 1, delay, task="baseline")
        assert queues[0].qsize() == queues[1].qsize()
        while queues[0].qsize():
            before, after = [queue.get_with_provenance() for queue in queues]
            torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
            assert before[1:] == after[1:]


@pytest.mark.skipif(os.environ.get("SALIENCY_COMPILE_TEST") != "1", reason="opt-in CUDA/Inductor test")
@pytest.mark.parametrize("kind", ["pi0", "pi05", "smolvla"])
@torch.no_grad()
def test_compile_default_sampling(kind, monkeypatch):
    import time

    old, new = models(kind, monkeypatch, device="cuda")
    old.eval()
    new.eval()
    kw = inputs(kind, "cuda")
    expected = old.sample_actions(**kw)
    eager = new.sample_actions(**kw)
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    compiled = torch.compile(new.sample_actions, backend="inductor", mode="max-autotune", fullgraph=True)
    torch.compiler.cudagraph_mark_step_begin()
    actual = compiled(**kw).clone()
    torch.cuda.synchronize()
    cold = time.perf_counter() - start
    # Compare the old compiled path too: a pre-existing compiler discrepancy must
    # be distinguished from a regression introduced by the saliency changes.
    old_compiled = torch.compile(old.sample_actions, backend="inductor", mode="max-autotune", fullgraph=True)
    torch.compiler.cudagraph_mark_step_begin()
    old_actual = old_compiled(**kw).clone()
    print(
        f"{kind}: old_compiled_vs_eager={(old_actual - expected).abs().max().item():.9g} "
        f"new_compiled_vs_eager={(actual - expected).abs().max().item():.9g}"
    )
    # Regression means old-vs-new under the same compiled entry. The miniature
    # pre-saliency SmolVLA already differs between eager and Inductor; logging
    # that discrepancy is useful, but it is not a saliency regression.
    torch.testing.assert_close(actual, old_actual, atol=1e-5, rtol=1e-5)
    start = time.perf_counter()
    for _ in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        compiled(**kw)
    torch.cuda.synchronize()
    print(f"{kind}: Inductor fullgraph cold={cold:.3f}s warm={(time.perf_counter() - start) / 3:.6f}s")
    torch._dynamo.reset()


def test_pi05_trained_rtc_explainers_with_miniature_policy(monkeypatch):
    """Exercise the trained hard-prefix success path; this is not a trained checkpoint."""
    import numpy as np

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05 import modeling_pi05
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.saliency.pi import PI05Explainer
    from lerobot.rollout.saliency.pi_attention import PI05AttentionExplainer
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    miniature(modeling_pi05, monkeypatch)
    camera = "observation.images.test"
    config = PI05Config(
        device="cpu",
        dtype="float32",
        image_resolution=(28, 28),
        chunk_size=4,
        n_action_steps=4,
        max_action_dim=4,
        max_state_dim=4,
        num_inference_steps=2,
        rtc_training_max_delay=2,
        rtc_config=RTCConfig(enabled=True, mode="trained", execution_horizon=2),
        input_features={camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
    )
    torch.manual_seed(419)
    policy = modeling_pi05.PI05Policy(config).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    batch = {
        camera: torch.rand(1, 3, 28, 28),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 64, (1, 3)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
    }
    noise = torch.randn(1, 4, 4)
    previous = torch.randn(2, 4)
    kwargs = {"inference_delay": 2, "prev_chunk_left_over": previous}

    with torch.no_grad():
        direct = policy.predict_action_chunk(batch, noise=noise, **kwargs)
    torch.testing.assert_close(direct[0, :2], previous, atol=0, rtol=0)
    target_kwargs = {**kwargs, "target_action_index": 2}
    replay_batch = {key: value.clone().requires_grad_(key == camera) for key, value in batch.items()}
    replay = PI05Explainer(policy)._sample_with_grad(replay_batch, noise, target_kwargs)
    torch.testing.assert_close(replay.detach(), direct, atol=1e-5, rtol=0)
    grad_maps = PI05Explainer(policy).compute_saliency(batch, noise, target_kwargs)
    assert grad_maps[camera].shape == (28, 28)
    assert np.isfinite(grad_maps[camera]).all()
    attention = PI05AttentionExplainer(policy)
    attn_maps = attention.compute_saliency(batch, noise, target_kwargs)
    assert attn_maps[camera].shape == (2, 2)
    assert np.isfinite(attn_maps[camera]).all()
    assert sum(attention.last_budget.values()) == pytest.approx(1, abs=1e-5)
    attention.close()


@pytest.mark.parametrize("kind", ["pi0", "pi05"])
@pytest.mark.parametrize("method", ["attention", "grad_x_input"])
@pytest.mark.parametrize("cross_device", [False, True])
def test_pi_replay_moves_full_leftover_and_encodes_prefix_once(kind, method, cross_device, monkeypatch):
    """Real miniature policies; full-shaped leftovers bypass RTC's padding branch."""
    import numpy as np

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.saliency import get_explainer_cls
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    if cross_device and torch.cuda.device_count() < 2:
        pytest.skip("cross-device regression requires two CUDA devices")
    source = "cuda:0" if cross_device else "cpu"
    target = "cuda:1" if cross_device else "cpu"
    module = importlib.import_module(f"lerobot.policies.{kind}.modeling_{kind}")
    miniature(module, monkeypatch)
    camera = "observation.images.test"
    config = getattr(module, kind.upper() + "Config")(
        device=target,
        dtype="float32",
        image_resolution=(28, 28),
        chunk_size=4,
        n_action_steps=4,
        max_action_dim=4,
        max_state_dim=4,
        num_inference_steps=2,
        rtc_config=RTCConfig(enabled=True, execution_horizon=4),
        input_features={
            camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
    )
    torch.manual_seed(419)
    policy = getattr(module, kind.upper() + "Policy")(config).to(target).eval()
    policy.requires_grad_(False)
    batch = {
        camera: torch.rand(1, 3, 28, 28, device=source),
        OBS_STATE: torch.rand(1, 4, device=source),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 64, (1, 3), device=source),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool, device=source),
    }
    noise = torch.randn(1, 4, 4, device=source)
    previous = torch.randn(4, 4, device=source)
    saved_previous = previous.clone()
    kwargs = {"inference_delay": 1, "prev_chunk_left_over": previous, "target_action_index": 2}
    calls = []
    embed = policy.model.embed_prefix

    def counted_embed(*args, **kwargs):
        calls.append(1)
        return embed(*args, **kwargs)

    monkeypatch.setattr(policy.model, "embed_prefix", counted_embed)
    denoise = policy.rtc_processor.denoise_step

    def checked_denoise(*args, **kwargs):
        assert kwargs["prev_chunk_left_over"].device == kwargs["x_t"].device
        return denoise(*args, **kwargs)

    monkeypatch.setattr(policy.rtc_processor, "denoise_step", checked_denoise)
    explainer = get_explainer_cls(kind, method)(policy)
    maps = explainer.compute_saliency(batch, noise, kwargs)
    assert len(calls) == 1
    assert np.isfinite(maps[camera]).all()
    assert maps[camera].shape == ((2, 2) if method == "attention" else (28, 28))
    torch.testing.assert_close(previous, saved_previous, atol=0, rtol=0)
    assert previous.device == torch.device(source)
    if method == "attention":
        explainer.close()
