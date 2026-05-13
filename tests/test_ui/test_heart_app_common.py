from checkmaite.ui._common.heart_app_common import HeartBaseApp


def test_heart_base_app_preserves_selection_stage_and_view_invariants() -> None:
    app = HeartBaseApp(task="object_detection")

    app.add_test_stage_callback(None)
    assert not app.attack_stages

    app.patch_attack_config = True
    app.enforce_patch_attack_type()
    app.pgd_attack_config = True
    app.enforce_patch_attack_type()
    assert sum([app.patch_attack_config, app.pgd_attack_config]) == 1

    app.strong_attack_config = True
    app.enforce_single_attack_strength()
    app.weak_attack_config = True
    app.enforce_single_attack_strength()
    assert sum([app.strong_attack_config, app.weak_attack_config]) == 1

    app.add_test_stage_callback(None)
    app._run_export()
    assert len(app.attack_stages) == len(app.output_test_stages) == 1

    app.clear_test_stage_callback(None)
    assert not app.attack_stages
