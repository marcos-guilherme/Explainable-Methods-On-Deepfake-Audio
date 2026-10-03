from jmds_prepare.core.two_source import TwoSourceArtifactNames as CoreNames
from jmds_prepare.profiles.portuguese import PORTUGUESE_ARTIFACT_NAMES, PORTUGUESE_PROFILE
from jmds_prepare.storage.metadata_layout import PortugueseMetadataLayout
from jmds_prepare.storage.two_source_metadata_layout import (
    TwoSourceArtifactNames,
    TwoSourceMetadataLayout,
)

NAMES = TwoSourceArtifactNames(
    pristine_manifest="coraa_metadata.csv",
    generated_manifest="jmds_mlaad_generated_metadata.csv",
    pristine_profile="coraa_metadata_profile.json",
    generated_profile="jmds_mlaad_generated_metadata_profile.json",
    summary="portuguese_metadata_summary.json",
    provenance="portuguese_metadata_provenance.json",
)


def test_two_source_artifact_names_is_canonical_in_core():
    assert TwoSourceArtifactNames is CoreNames
    assert TwoSourceArtifactNames.__module__ == "jmds_prepare.core.two_source"


def test_portuguese_profile_exposes_artifact_names():
    assert PORTUGUESE_PROFILE.artifact_names is PORTUGUESE_ARTIFACT_NAMES
    assert PORTUGUESE_PROFILE.artifact_names.pristine_manifest == "coraa_metadata.csv"
    assert (
        PORTUGUESE_PROFILE.artifact_names.generated_manifest
        == "jmds_mlaad_generated_metadata.csv"
    )


def test_two_source_layout_exposes_manifest_and_report_dirs(tmp_path):
    layout = TwoSourceMetadataLayout(tmp_path / "out", NAMES)
    assert layout.manifests_dir == tmp_path / "out" / "manifests"
    assert layout.reports_dir == tmp_path / "out" / "reports"


def test_two_source_layout_resolves_six_paths(tmp_path):
    layout = TwoSourceMetadataLayout(tmp_path / "out", NAMES)
    assert layout.pristine_metadata_csv == (
        tmp_path / "out" / "manifests" / "coraa_metadata.csv"
    )
    assert layout.generated_metadata_csv == (
        tmp_path / "out" / "manifests" / "jmds_mlaad_generated_metadata.csv"
    )
    assert layout.pristine_metadata_profile_json == (
        tmp_path / "out" / "reports" / "coraa_metadata_profile.json"
    )
    assert layout.generated_metadata_profile_json == (
        tmp_path
        / "out"
        / "reports"
        / "jmds_mlaad_generated_metadata_profile.json"
    )
    assert layout.summary_json == (
        tmp_path / "out" / "reports" / "portuguese_metadata_summary.json"
    )
    assert layout.provenance_json == (
        tmp_path / "out" / "reports" / "portuguese_metadata_provenance.json"
    )


def test_portuguese_layout_paths_match_generic_wrapper(tmp_path):
    """PortugueseMetadataLayout alias properties match the generic layout."""
    output_root = tmp_path / "prepared"
    generic = TwoSourceMetadataLayout(output_root, NAMES)
    portuguese = PortugueseMetadataLayout(output_root)

    assert portuguese.coraa_metadata_csv == generic.pristine_metadata_csv
    assert (
        portuguese.jmds_mlaad_generated_metadata_csv
        == generic.generated_metadata_csv
    )
    assert (
        portuguese.coraa_metadata_profile_json
        == generic.pristine_metadata_profile_json
    )
    assert (
        portuguese.jmds_mlaad_generated_metadata_profile_json
        == generic.generated_metadata_profile_json
    )
    assert (
        portuguese.portuguese_metadata_summary_json == generic.summary_json
    )
    assert (
        portuguese.portuguese_metadata_provenance_json
        == generic.provenance_json
    )
