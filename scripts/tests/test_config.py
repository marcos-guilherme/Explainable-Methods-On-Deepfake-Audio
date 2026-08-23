from brspeech_xai.config import RunConfig, load_config


def test_load_default_and_override(tmp_path):
    yaml_text = """
run_name: default
seed: 42
device: cpu
data:
  dataset_id: AKCIT-Deepfake/BRSpeech-DF
  loader: auto
  train_split: train
  eval_split: test
  n_train_per_class: 1500
  n_test_per_class: 1500
audio: {sample_rate: 16000, num_samples: 64600}
model: {checkpoint: nii-yamagishilab/mms-300m-anti-deepfake, spoof_index: 0}
bands: {n_bands: 8, f_min: 20.0, f_max: 7900.0}
occlusion: {per_quadrant: 150}
association: {top_n: 10}
output_dir: results
"""
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(yaml_text)

    cfg = load_config(cfg_file, overrides=["data.n_test_per_class=50", "seed=7",
                                           "bands.n_bands=16"])
    assert isinstance(cfg, RunConfig)
    assert cfg.seed == 7
    assert cfg.data.n_test_per_class == 50
    assert cfg.data.n_train_per_class == 1500
    assert cfg.audio.num_samples == 64600
    assert cfg.bands.n_bands == 16  # grade compartilhada, override por CLI


def test_adapt_defaults_and_override(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("run_name: r\nseed: 1\ndevice: cpu\n")
    cfg = load_config(cfg_file)
    assert cfg.adapt.cross_fit is False  # padrão = holdout (train->test)
    assert cfg.adapt.cv_folds == 5
    assert cfg.adapt.head == "logistic"
    assert cfg.data.analysis_split == "train"
    assert cfg.data.n_analysis_per_class == 10000
    assert cfg.bands.n_bands == 8  # default preserva a run de referência
    assert cfg.model.encoder == "xlsr_fairseq"  # default preserva o XLS-R

    cfg2 = load_config(cfg_file, overrides=["adapt.head=mlp", "adapt.cross_fit=true",
                                            "data.n_analysis_per_class=20",
                                            "model.encoder=hf_ssl"])
    assert cfg2.adapt.head == "mlp"
    assert cfg2.adapt.cross_fit is True
    assert cfg2.data.n_analysis_per_class == 20
    assert cfg2.model.encoder == "hf_ssl"  # trocável por CLI (registro futuro)


def test_config_hash_is_stable_and_sensitive(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("run_name: r\nseed: 1\ndevice: cpu\n")
    a = load_config(cfg_file)
    b = load_config(cfg_file)
    c = load_config(cfg_file, overrides=["seed=2"])
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()
