from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.common.vla_utils import resize_with_pad_torch  # noqa: E402
from lerobot.policies.pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
from lerobot.transforms import ImageTransformConfig, ImageTransforms, ImageTransformsConfig  # noqa: E402
from lerobot.transforms.openpi import OpenPIImageAugmentation, OpenPIImageAugmentationConfig  # noqa: E402
from lerobot.transforms.openpi import _augment_openpi_images  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402


def _bare_policy(config):
    policy = PI05Policy.__new__(PI05Policy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.anchor = nn.Parameter(torch.zeros(()))
    return policy


def _openpi_transforms(**overrides):
    return ImageTransforms(
        ImageTransformsConfig(
            enable=True,
            max_num_transforms=1,
            tfs={"openpi": ImageTransformConfig(type="OpenPIImageAugmentation", kwargs=overrides)},
        )
    )


def test_openpi_augmentation_defaults_and_switch():
    config = ImageTransformsConfig()
    assert config.enable is False
    defaults = OpenPIImageAugmentation().config
    assert defaults.crop_scale == 0.95
    assert defaults.rotation_degrees == 5.0
    assert defaults.color_probability == 0.5
    assert defaults.brightness == 0.3
    assert defaults.contrast == 0.4
    assert defaults.hue == 0.1
    assert PI05Config().train_on_padded_action_dims is False

    image = torch.rand(3, 16, 16)
    disabled = ImageTransforms(
        ImageTransformsConfig(enable=False, tfs={"openpi": ImageTransformConfig(type="OpenPIImageAugmentation")})
    )
    torch.testing.assert_close(disabled(image), image)


def test_openpi_augmentation_is_per_image_and_camera_aware():
    image = torch.full((3, 224, 224), 0.5)
    batch = torch.stack([image] * 8)
    augment = _openpi_transforms()

    torch.manual_seed(7)
    augmented_top = torch.stack([augment(item, camera_key="observation.images.top") for item in batch])
    torch.manual_seed(7)
    augmented_wrist = torch.stack([augment(item, camera_key="observation.images.left_wrist") for item in batch])

    assert augmented_top.shape == batch.shape
    assert augmented_wrist.shape == batch.shape
    assert torch.isfinite(augmented_top).all()
    assert augmented_top.min() >= 0 and augmented_top.max() <= 1
    assert torch.all(augmented_top.flatten(1).std(dim=1) > 0)
    assert torch.all(augmented_wrist.flatten(1).std(dim=1) < 1e-6)
    assert any(not torch.equal(augmented_top[0], item) for item in augmented_top[1:])


def test_openpi_transform_resizes_padded_uint8_and_keeps_history_frames_independent():
    image = torch.full((2, 3, 120, 240), 128, dtype=torch.uint8)
    image[0, :, 30:90, 60:180] = 255
    image[1, :, 30:90, 60:180] = 255
    augment = _openpi_transforms(color_probability=0.0)
    torch.manual_seed(4)
    output = augment(image, camera_key="observation.images.top")
    assert output.shape == (2, 3, 224, 224)
    assert output.dtype == torch.float32
    assert output.min() >= 0 and output.max() <= 1
    assert not torch.equal(output[0], output[1])


def test_openpi_dataset_resize_matches_pi05_model_resize_when_jitter_is_neutral():
    image = torch.rand(3, 120, 240)
    augment = _openpi_transforms(
        crop_scale=1.0,
        rotation_degrees=0.0,
        color_probability=0.0,
    )
    output = augment(image, camera_key="observation.images.top")
    expected = resize_with_pad_torch(image.unsqueeze(0), 224, 224)[0]
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=0)


def test_image_augmentation_is_absent_from_pi05_model_preprocessing():
    key = "observation.images.top"
    image = torch.rand(2, 3, 16, 16)
    config = SimpleNamespace(image_features=[key], image_resolution=(16, 16))
    policy = _bare_policy(config)
    processed, _ = policy._preprocess_images({key: image})
    torch.testing.assert_close(processed[0], image * 2 - 1)
    policy.eval()
    eval_processed, _ = policy._preprocess_images({key: image})
    torch.testing.assert_close(eval_processed[0], image * 2 - 1)


def test_openpi_crop_zooms_in_after_resize():
    image = torch.zeros((1, 224, 224, 3))
    image[:, 56:168, 56:168] = 1.0

    with patch.object(torch, "rand", side_effect=lambda *shape, device=None: torch.full(shape, 0.5, device=device)):
        augmented = _augment_openpi_images(image, "observation.images.top", OpenPIImageAugmentationConfig())

    assert augmented.sum() > image.sum() * 1.05


def test_openpi_augmentation_parameters_change_transform():
    image = torch.zeros((1, 224, 224, 3))
    image[:, 56:168, 56:168] = 1.0
    neutral = OpenPIImageAugmentationConfig(crop_scale=1.0, rotation_degrees=0.0, color_probability=0.0)
    crop = OpenPIImageAugmentationConfig(crop_scale=0.90, rotation_degrees=0.0, color_probability=0.0)
    with patch.object(torch, "rand", side_effect=lambda *shape, device=None: torch.full(shape, 0.5, device=device)):
        unchanged = _augment_openpi_images(image, "observation.images.top", neutral)
        enlarged = _augment_openpi_images(image, "observation.images.top", crop)

    torch.testing.assert_close(unchanged, image, atol=1e-5, rtol=0)
    assert enlarged.sum() > unchanged.sum() * 1.1


def test_openpi_color_probability_and_brightness_are_configurable():
    image = torch.full((1, 16, 16, 3), 0.5)
    config = OpenPIImageAugmentationConfig(color_probability=0.0, brightness=0.3, contrast=0.0, hue=0.0)
    with patch.object(torch, "rand", side_effect=lambda *shape, device=None: torch.full(shape, 0.75, device=device)):
        no_color = _augment_openpi_images(image, "observation.images.wrist", config)
        config.color_probability = 1.0
        brighter = _augment_openpi_images(image, "observation.images.wrist", config)

    torch.testing.assert_close(no_color, image)
    assert torch.all(brighter > no_color)


class _ElementwiseLossModel:
    def __init__(self):
        self.loss_values = nn.Parameter(torch.ones(32))

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def sample_time(self, batch_size, device):
        return torch.zeros(batch_size, device=device)

    def forward(self, *args, **kwargs):
        actions = args[4]
        return self.loss_values.view(1, 1, -1).expand(actions.shape[0], actions.shape[1], -1)


def _loss_policy(train_on_padded_action_dims):
    config = SimpleNamespace(
        max_action_dim=32,
        output_features={ACTION: SimpleNamespace(shape=(12,))},
        use_proprioceptive_memory=False,
        rtc_training_max_delay=0,
        train_on_padded_action_dims=train_on_padded_action_dims,
    )
    policy = _bare_policy(config)
    policy.model = _ElementwiseLossModel()
    policy._preprocess_images = lambda batch: ([], [])
    batch = {
        ACTION: torch.zeros(1, 2, 12),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
    }
    return policy, batch


def test_action_loss_switch_controls_padding_dimensions_and_gradients():
    padded_policy, padded_batch = _loss_policy(True)
    padded_loss, padded_info = padded_policy(padded_batch)
    padded_loss.backward()

    legacy_policy, legacy_batch = _loss_policy(False)
    legacy_loss, legacy_info = legacy_policy(legacy_batch)
    legacy_loss.backward()

    assert padded_info["loss"] == pytest.approx(1.0)
    assert legacy_info["loss"] == pytest.approx(1.0)
    assert torch.all(padded_policy.model.loss_values.grad[12:] > 0)
    assert torch.all(legacy_policy.model.loss_values.grad[12:] == 0)
    assert padded_policy.model.loss_values.grad[0] == pytest.approx(1 / 32)
    assert legacy_policy.model.loss_values.grad[0] == pytest.approx(1 / 12)


def test_merge_quantiles_preserves_mean_std_and_other_existing_stats(tmp_path):
    from examples.pi05.prepare_train_quantile_stats import merge_quantile_stats
    from lerobot.datasets.io_utils import load_stats, write_stats

    existing = {
        "observation.state": {"mean": [1.0], "std": [2.0], "min": [0.0], "count": [10]},
        "action": {"mean": [3.0], "std": [4.0]},
    }
    merged = merge_quantile_stats(
        existing,
        {
            "observation.state": {"q01": torch.tensor([-1.0]), "q99": torch.tensor([5.0])},
            "action": {"q01": torch.tensor([-2.0]), "q99": torch.tensor([6.0])},
        },
    )

    assert merged["observation.state"] == {
        "mean": [1.0], "std": [2.0], "min": [0.0], "count": [10], "q01": [-1.0], "q99": [5.0]
    }
    assert merged["action"] == {"mean": [3.0], "std": [4.0], "q01": [-2.0], "q99": [6.0]}
    assert existing["observation.state"] == {"mean": [1.0], "std": [2.0], "min": [0.0], "count": [10]}

    (tmp_path / "meta").mkdir()
    write_stats(merged, tmp_path)
    restored = load_stats(tmp_path)
    np.testing.assert_array_equal(restored["observation.state"]["mean"], [1.0])
    np.testing.assert_array_equal(restored["observation.state"]["std"], [2.0])
    np.testing.assert_array_equal(restored["observation.state"]["q01"], [-1.0])
    np.testing.assert_array_equal(restored["action"]["mean"], [3.0])
    np.testing.assert_array_equal(restored["action"]["q99"], [6.0])


def test_quantile_preparation_uses_native_eval_split(tmp_path, lerobot_dataset_factory):
    from examples.pi05.prepare_train_quantile_stats import prepare_train_quantile_stats
    from lerobot.datasets.io_utils import load_stats, write_stats
    from tests.fixtures.constants import DUMMY_REPO_ID

    root = tmp_path / "dataset"
    lerobot_dataset_factory(root=root, total_episodes=3, total_frames=15, use_videos=False)
    original_stats = load_stats(root)
    original_stats["observation.state"] = {"mean": np.array([0.5]), "std": np.array([0.2])}
    write_stats(original_stats, root)
    seen_episodes = []

    def fake_episode_arrays(dataset, episode, **kwargs):
        seen_episodes.append(episode)
        values = np.array([[float(episode)], [float(episode) + 0.5]])
        return {
            "observation.state": (values, 2),
            "action": (values, 2),
        }

    with patch("examples.pi05.prepare_train_quantile_stats.collect_episode_arrays", side_effect=fake_episode_arrays):
        result = prepare_train_quantile_stats(DUMMY_REPO_ID, root, eval_split=1 / 3)

    assert result["train_episodes"] == 2
    assert seen_episodes == [0, 1]
    restored = load_stats(root)
    np.testing.assert_array_equal(
        restored["observation.state"]["mean"], original_stats["observation.state"]["mean"]
    )
    assert "q01" in restored["observation.state"]
    assert "q99" in restored["action"]
