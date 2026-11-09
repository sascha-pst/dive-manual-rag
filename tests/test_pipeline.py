import pytest

from diverag.answer import answer
from diverag.chunk import fixed_chunks, section_chunks
from diverag.evaluate import compare, load_questions
from diverag.index import Index
from diverag.parse import clean_pages, load_sample, split_sections
from diverag.retrieve import rrf
from diverag.sources import LicenseError, load_manifest


@pytest.fixture(scope="module")
def sections():
    return load_sample()


@pytest.fixture(scope="module")
def ix(sections):
    return Index(section_chunks(sections))


# ---- sources ----
def test_manifest_loads():
    assert {s.id for s in load_manifest()} >= {"usn-dm-r7a", "noaa-ndssm"}


def test_license_gate_blocks(tmp_path):
    bad = tmp_path / "s.toml"
    bad.write_text(
        '[[source]]\nid="x"\ntitle="t"\npublisher="p"\nlicense="copyrighted"\n'
        'revision="r"\neffective="e"\nurl="u"\npath="p"\n'
    )
    with pytest.raises(LicenseError):
        load_manifest(bad)


# ---- parse ----
def test_cleaning_removes_headers_and_hyphens():
    pages = [
        f"DIVING MANUAL VOL 1\n2-{i} Pressure Topic\nPage {i} says absolute pres-\nsure is the sum.\n"
        f"More body text on page {i}.\nEven more body text.\nChange A"
        for i in range(1, 4)
    ]
    text = clean_pages(pages)
    assert "DIVING MANUAL" not in text and "Change A" not in text
    assert "Page 2 says absolute pressure is the sum." in text
    assert [s.section for s in split_sections(text, "t", "r")] == ["2-1", "2-2", "2-3"]


# ---- chunk ----
def test_section_chunks_never_mix(sections):
    assert all(len(c.sections) == 1 for c in section_chunks(sections, 40, 10))


def test_fixed_chunks_do_mix(sections):
    assert any(len(c.sections) > 1 for c in fixed_chunks(sections, 90, 10))


# ---- retrieve ----
def test_rrf_prefers_agreement():
    assert rrf([[1, 2, 3], [2, 1, 3]], [0.5, 0.5])[-1] == 3


def test_semantic_finds_slang(ix):
    a = answer(ix, "what causes the bends")
    assert a.route == "explain" and "§3-4" in a.citations


# ---- routing ----
def test_lookup_is_computed(ix):
    a = answer(ix, "PO2 of 32% oxygen at 99 fsw")
    assert a.route == "lookup" and "4.00 ata" in a.text and "1.28 ata" in a.text


def test_personal_advice_is_referred(ix):
    a = answer(ix, "can I fly 6 hours after my dive")
    assert a.route == "advice" and "§4-2" in a.citations


def test_off_topic_is_refused(ix):
    assert answer(ix, "best dive sites in Bonaire").route == "refuse"


# ---- eval ----
def test_hybrid_beats_bm25(ix):
    s = compare(ix, load_questions())
    assert s["hybrid+rerank"].recall >= s["bm25"].recall
    assert s["hybrid+rerank"].recall >= 0.9


def test_fixed_chunking_lowers_purity(sections):
    q = load_questions()
    sec = compare(Index(section_chunks(sections)), q)["hybrid"].purity
    fix = compare(Index(fixed_chunks(sections)), q)["hybrid"].purity
    assert fix < sec
