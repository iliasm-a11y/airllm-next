"""Pure boundary policy; no model, subprocess or native API is used."""
import pytest
from research.glm53_batched_prefill.prefix_boundary import choose_boundary, report_boundary


@pytest.mark.parametrize("microbatch,expected", [(8, 208), (16, 208), (32, 192), (64, 192), (128, 128)])
def test_common_native_ids_choose_a_grid_boundary(microbatch, expected):
    common = list(range(211))
    assert choose_boundary([common + [1001], common + [1002]], microbatch) == expected


def test_explicit_common_boundary_allows_a_cross_batch_comparison():
    common = list(range(211))
    prompts = [common + [1001], common + [1002]]
    assert choose_boundary(prompts, 8, 128) == choose_boundary(prompts, 128, 128) == 128


@pytest.mark.parametrize("requested", [True, 0, 208, 224])
def test_invalid_or_unaligned_requested_boundary_is_refused(requested):
    with pytest.raises(ValueError):
        choose_boundary([list(range(211)) + [999], list(range(211)) + [998]], 32, requested)


def test_snapshot_never_consumes_the_entire_prompt():
    assert choose_boundary([list(range(16)), list(range(16))], 8) == 8


def test_new_report_cannot_claim_an_unaligned_boundary():
    with pytest.raises(ValueError):
        report_boundary({"microbatch": 32, "prefix_boundary": 208, "boundary_policy": "native-lcp-aligned-v1"})
    assert report_boundary({"microbatch": 32, "prefix_boundary": 192, "boundary_policy": "native-lcp-aligned-v1"}) == 192


def test_legacy_receipts_keep_their_original_boundary():
    assert report_boundary({"microbatch": 8}) == 208
