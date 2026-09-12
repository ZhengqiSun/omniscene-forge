#!/usr/bin/env python3
"""CPU contract tests for the isolated LingBot Fast v2 runtime foundation."""

from __future__ import annotations

from pathlib import Path
import unittest

from lingbot_fast_v2_runtime_v0 import (
    DEFAULT_SOURCE_ROOT,
    PUBLIC_CHUNK_SIZE,
    PUBLIC_LOCAL_ATTN_SIZE,
    PUBLIC_SHIFT,
    PUBLIC_SINK_SIZE,
    PUBLIC_TIMESTEP_INDICES,
    cache_index_expectations,
    cache_shapes,
    canonical_latent_hw,
    clean_commit_lifecycle,
    derive_public_contract,
    derive_scheduler_selection,
    frame_behavior,
    load_official_scheduler_class,
    official_non_overlapping_chunks,
)


class LingBotFastV2RuntimeContractTests(unittest.TestCase):
    source_root = Path(DEFAULT_SOURCE_ROOT)

    def test_official_scheduler_is_authoritative_for_timesteps_and_sigmas(self) -> None:
        public = derive_public_contract(self.source_root)
        selection = derive_scheduler_selection(
            self.source_root,
            num_train_timesteps=public["num_train_timesteps"],
            shift=public["shift"],
            timestep_indices=public["outer_generate_timestep_indices"],
        )
        scheduler_class = load_official_scheduler_class(self.source_root)
        scheduler = scheduler_class(
            num_train_timesteps=public["num_train_timesteps"],
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(public["num_train_timesteps"], shift=PUBLIC_SHIFT)

        self.assertEqual(public["outer_generate_timestep_indices"], list(PUBLIC_TIMESTEP_INDICES))
        for item in selection["selected"]:
            index = item["outer_index"]
            self.assertEqual(item["timestep"], int(scheduler.timesteps[index].item()))
            self.assertEqual(item["sigma"], float(scheduler.sigmas[index].item()))

    def test_81_frames_exposes_usable_latent_and_output_reduction(self) -> None:
        behavior = frame_behavior(81)
        self.assertEqual(behavior.normalized_requested_frames, 81)
        self.assertEqual(behavior.pre_chunk_latent_frames, 21)
        self.assertEqual(behavior.usable_latent_frames, 20)
        self.assertEqual(behavior.dropped_latent_frames, 1)
        self.assertEqual(behavior.output_frames, 77)
        self.assertEqual(behavior.dropped_input_to_output_frames, 4)
        self.assertIsNotNone(behavior.warning)

    def test_non_overlapping_chunks_and_clean_commit(self) -> None:
        frame_seqlen = 58 * 104 // 4
        chunks = official_non_overlapping_chunks(20, frame_seqlen=frame_seqlen)
        self.assertEqual(
            [(item.latent_start, item.latent_end) for item in chunks],
            [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)],
        )
        self.assertEqual([item.current_start for item in chunks], [0, 6032, 12064, 18096, 24128])
        public = derive_public_contract(self.source_root)
        scheduler = derive_scheduler_selection(
            self.source_root,
            num_train_timesteps=public["num_train_timesteps"],
            shift=public["shift"],
            timestep_indices=public["outer_generate_timestep_indices"],
        )
        lifecycle = clean_commit_lifecycle(
            [item["timestep"] for item in scheduler["selected"]]
        )
        self.assertEqual([item["phase"] for item in lifecycle], ["denoise"] * 4 + ["clean_commit"])
        self.assertEqual(lifecycle[-1]["timestep"], 0)
        self.assertTrue(lifecycle[-1]["commits_clean_x0"])

    def test_cache_shapes_and_window_roll_indices(self) -> None:
        latent_hw = canonical_latent_hw(
            image_height=480,
            image_width=832,
            max_area=480 * 832,
        )
        self.assertEqual(latent_hw, (58, 104))
        shapes = cache_shapes(
            batch_size=1,
            latent_height=latent_hw[0],
            latent_width=latent_hw[1],
            dim=5120,
            num_heads=40,
            num_layers=40,
            sequence_parallel_size=8,
        )
        self.assertEqual(shapes["frame_seqlen"], 1508)
        self.assertEqual(shapes["self_kv_per_layer"], [1, 27144, 5, 128])
        self.assertEqual(shapes["cross_kv_per_layer"], [1, 512, 40, 128])

        chunks = official_non_overlapping_chunks(20, frame_seqlen=shapes["frame_seqlen"])
        indices = cache_index_expectations(
            chunks,
            frame_seqlen=shapes["frame_seqlen"],
            local_attn_size=PUBLIC_LOCAL_ATTN_SIZE,
            sink_size=PUBLIC_SINK_SIZE,
        )
        frame_seqlen = shapes["frame_seqlen"]
        self.assertEqual([item.global_end_after // frame_seqlen for item in indices], [4, 8, 12, 16, 20])
        self.assertEqual([item.local_end // frame_seqlen for item in indices], [4, 8, 12, 16, 18])
        self.assertEqual(indices[-1].evicted_tokens // frame_seqlen, 2)
        self.assertEqual(indices[-1].local_start // frame_seqlen, 14)

    def test_resolved_public_values_are_explicit(self) -> None:
        public = derive_public_contract(self.source_root)
        self.assertEqual(public["chunk_size"], PUBLIC_CHUNK_SIZE)
        self.assertEqual(public["shift"], PUBLIC_SHIFT)
        self.assertEqual(public["local_attn_size"], PUBLIC_LOCAL_ATTN_SIZE)
        self.assertEqual(public["sink_size"], PUBLIC_SINK_SIZE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
