"""Display-free stand-ins for the Tk state the studio code reads."""

from __future__ import annotations

import queue
from unittest import mock

import config
import gui


class FakeVar:
    """Minimal tk.Variable replacement: get() and set() only."""

    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def studio_app(mode: str = "regular", *, count: int = 12, seed: int = 42):
    """Return a bare gui.App wired with fake variables for the Generate tab."""
    app = object.__new__(gui.App)
    app.q = queue.Queue()
    app.count_var = FakeVar(str(count))
    app.seed_var = FakeVar(str(seed))
    app.dataset_var = FakeVar("(next)")
    app.names_var = FakeVar(config.NAMES_VERSION)
    app.mode_by_label = {label: key for key, label in config.SAMPLE_MODES.items()}
    app.sample_mode_var = FakeVar(config.SAMPLE_MODES[mode])
    app.font_style_options = {"All fonts": "all", "Cursive only": "cursive"}
    app.font_style_var = FakeVar("All fonts")
    app.cursive_group_var = FakeVar("All cursive")
    app.specific_font_var = FakeVar("All in group")
    app.specific_font_map = {"All in group": ""}
    app.degradation_by_label = {
        label: key for key, label in config.DEGRADATION_PROFILES.items()
    }
    app.degradation_var = FakeVar(
        config.DEGRADATION_PROFILES[config.DEFAULT_DEGRADATION_PROFILE]
    )
    app.clipping_by_label = {
        label: key for key, label in config.EDGE_CLIPPING_OPTIONS.items()
    }
    app.edge_clipping_var = FakeVar(
        config.EDGE_CLIPPING_OPTIONS[config.DEFAULT_EDGE_CLIPPING]
    )
    app.dev_sliders = {
        name: FakeVar(spec.default) for name, spec in gui.SLIDER_SPECS.items()
    }
    app.dev_labels = {name: FakeVar("") for name in gui.SLIDER_SPECS}
    app.dev_scales = {}
    app._slider_updating = False
    app._run_control_states = []
    app.real_var = FakeVar(False)
    app.zip_var = FakeVar(False)
    app.split_train_var = FakeVar(gui._split_percent(config.SYNTH_TRAIN_FRAC))
    app.split_val_var = FakeVar(gui._split_percent(config.SYNTH_VAL_FRAC))
    app.split_test_var = FakeVar(gui._split_percent(config.SYNTH_TEST_FRAC))
    app.field_weight_vars = {
        ftype: FakeVar("") for ftype in sorted(config.SUPPORTED_FIELD_TYPES)
    }
    app._reset_field_weights()
    app.workers_var = FakeVar("Auto (Max)")
    app.preset_var = FakeVar("")
    app.status = mock.Mock()
    app._update_preview = mock.Mock()
    app._on_font_style_change = mock.Mock()
    app._recipe_sections = {}
    return app
