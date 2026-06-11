"""Shared pytest fixtures for tpx3pipe tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def base_cfg():
    return {
        "seed": 42,
        "paths": {"pixel_data": "data/pixels.parquet", "tdc_data": "data/tdc.parquet",
                  "artifacts": "artifacts", "plots": "plots"},
        "tdc": {"period_override_ns": None, "missed_edge_threshold": 1.5},
        "clustering": {"dt_ns": 2000, "max_gap": 0, "time_criterion": "chain",
                       "safety_margin_ns": 200},
        "tof": {"cluster_time_mode": "t_min"},
        "labeling": {
            "n_peaks": 2,
            "peak_means_init_ns": [175.0, 250.0],
            "peak_stds_init_ns": [20.0, 20.0],
            "n_em_iter": 50,
            "confidence_threshold": 0.90,
        },
        "baseline": {"train_frac": 0.70, "val_frac": 0.15},
        "cnn": {"patch_size": 16, "batch_size": 16, "max_epochs": 2,
                "early_stopping_patience": 2, "learning_rate": 1e-3,
                "use_soft_labels": False, "augmentation": False},
        "synth": {
            "period_ns": 1785.0,
            "period_jitter_ns": 0.5,
            "duration_s": 0.003,      # short: ~1680 pulses → ~1000 clusters
            "missed_edge_rate": 0.02,
            "species": [
                {"name": "gamma", "rate_per_pulse": 0.5, "tof_mean_ns": 175.0,
                 "tof_std_ns": 10.0, "archetype": "dot"},
                {"name": "neutron", "rate_per_pulse": 0.3, "tof_mean_ns": 250.0,
                 "tof_std_ns": 15.0, "archetype": "blob"},
                {"name": "slow", "rate_per_pulse": 0.1, "tof_mean_ns": 2100.0,
                 "tof_std_ns": 100.0, "archetype": "curl"},
            ],
            "archetypes": {
                "dot": {"n_pixels_range": [1, 3], "tot_mean": 80, "tot_std": 15},
                "curl": {"n_pixels_range": [5, 15], "tot_mean": 120, "tot_std": 30},
                "blob": {"n_pixels_range": [4, 12], "tot_mean": 500, "tot_std": 100},
            },
            "intra_cluster_spread_ns": 50.0,
            "time_walk_delay_ns": 80.0,
            "time_walk_threshold_tot": 100,
            "shuffle_horizon_ns": 200000.0,
        },
    }


@pytest.fixture(scope="session")
def synth_data(base_cfg):
    from tpx3pipe.synth import generate
    pixel_df, tdc_df, truth_df = generate(base_cfg, seed=42)
    return pixel_df, tdc_df, truth_df
