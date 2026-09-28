from importlib.metadata import EntryPoint, entry_points

PACKAGE_NAME = "checkmaite"

# Model wrappers are advertised by modelmaite and the on-disk dataset classes by datamaite.
# CheckMAITE advertises what it still contains, including the Ray Serve clients.
EXPECTED_PROTOCOL_ENTRY_POINTS: dict[str, frozenset[str]] = {
    "maite.protocols.object_detection.Dataset": frozenset({"checkmaite_XaitkExplainableDetectionBaselineDataset"}),
    "maite.protocols.object_detection.Metric": frozenset(
        {
            "checkmaite_TorchODMetric",
            "checkmaite_TorchODMultiClassMap50",
        }
    ),
    "maite.protocols.image_classification.Metric": frozenset({"checkmaite_TorchICMulticlassMetric"}),
    "maite.protocols.object_detection.DataLoader": frozenset({"checkmaite_YoloDetectionDataLoader"}),
    "maite.protocols.image_classification.DataLoader": frozenset({"checkmaite_YoloClassificationDataLoader"}),
    "maite.protocols.object_detection.Model": frozenset({"checkmaite_RayObjectDetectionClient"}),
    "maite.protocols.image_classification.Model": frozenset({"checkmaite_RayImageClassificationClient"}),
}

# On-disk image-classification datasets live in datamaite (!612), so that group stays
# empty here. object_detection.Dataset is not delegated: the XAI baseline dataset is
# defined in this package.
DELEGATED_GROUPS = ("maite.protocols.image_classification.Dataset",)

EXPECTED_TASK_ENTRY_POINTS = frozenset(
    {
        "checkmaite_predict",
        "checkmaite_evaluate",
        "checkmaite_evaluate_from_predictions",
    }
)


def _checkmaite_entry_points(group: str) -> dict[str, EntryPoint]:
    selected = entry_points(group=group)
    return {ep.name: ep for ep in selected if ep.dist is not None and ep.dist.name == PACKAGE_NAME}


def test_protocol_entry_point_names_match_expected_set() -> None:
    seen: set[str] = set()
    for group, expected in EXPECTED_PROTOCOL_ENTRY_POINTS.items():
        names = set(_checkmaite_entry_points(group))
        assert names == expected, group
        overlap = seen & names
        assert not overlap, f"duplicate entry point names across groups: {overlap}"
        seen |= names


def test_delegated_groups_are_not_advertised_here() -> None:
    for group in DELEGATED_GROUPS:
        assert not _checkmaite_entry_points(group), group


def test_task_entry_point_names_match_expected_set() -> None:
    names = set(_checkmaite_entry_points("maite.tasks"))
    assert names == EXPECTED_TASK_ENTRY_POINTS
    overlap = names & {n for ns in EXPECTED_PROTOCOL_ENTRY_POINTS.values() for n in ns}
    assert not overlap


def test_every_advertised_entry_point_loads() -> None:
    for group, expected in EXPECTED_PROTOCOL_ENTRY_POINTS.items():
        loaded = _checkmaite_entry_points(group)
        for name in expected:
            assert name in loaded, (group, name)
            loaded[name].load()
    task_eps = _checkmaite_entry_points("maite.tasks")
    for name in EXPECTED_TASK_ENTRY_POINTS:
        assert name in task_eps, name
        task_eps[name].load()
