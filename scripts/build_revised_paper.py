"""Build BRACU_SJRC2026_FinalPaper.docx from BRACU_SJRC2026_Final_file.docx.

Format is preserved by construction: the original document is opened and only
text runs / table cells are edited in place. Styles, section/margins, image,
caption, spacing, alignment and table style are untouched. Same paragraph and
table count (table gains 2 scorer rows); body text kept near original length
for the one-page limit (swap, not add).
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "BRACU_SJRC2026_Final_file.docx"
DST = REPO / "BRACU_SJRC2026_FinalPaper_v5.docx"
FIG = REPO / "data/figures/method_pipeline_validated.png"
FIG_CAPTION = "Figure 1. Validated Methodological Pipeline for MoltBook Sentiment Dynamics."

INTRO = (
    "MoltBook is the first social network for autonomous AI agents. Prior work "
    "documents neutral, hub-dominated, weakly reciprocated discourse, leaving micro-dynamics "
    "unexamined. We analyze 1,241 parent\u2013child pairs across 49 threads: (RQ1) corpus distribution "
    "across authors and contexts; (RQ2) reply alignment versus de-escalation; (RQ3) sentiment trajectories "
    "across thread phases; (RQ4) robustness checks. All scorers are validated on a 250-comment "
    "human gold standard with double-annotated overlap (\u03ba reported), the primary scorer is selected "
    "empirically, and reply edges are sequential proxies, disclosed throughout."
)

RESULTS = (
    "Ensemble predictions are 54.8% neutral (95% CI 52.2\u201357.7), 39.0% positive, 6.2% negative. "
    "Variation is largest across authors (H = 308.7, \u03b5\u00b2 = .124); post/thread groupings coincide "
    "and are reported once (H = 110.9, \u03b5\u00b2 = .052). Sequential parent\u2013child pairs show no evidence "
    "of dependence (48.1% match; \u03c7\u00b2(4) = 5.00, p = .287; V = .045; permutation p = .087). Neutral "
    "self-transitions (57.3%) match the marginal rate (lift 1.02); negative-to-neutral transitions are 62.7% "
    "versus 56.2% (lift 1.11, n = 75). Early-to-late positive drift (29.2% \u2192 36.5%) is descriptive only "
    "(trend p = .32). On 246 gold labels (\u03ba = .393 on 50 paired), twitter-roberta and bertweet "
    "(macro-F1 .459) outperform the ensemble (.368; thresholds .05/.02, agree-or-neutral backoff). VADER "
    "alone inverts the ranking (70.2% positive): scorer selection must be empirical."
)

CONCLUSION = (
    "We find no evidence of sentiment contagion: neutral continuity reflects base rates, positive drift "
    "lacks inferential support, and authors show stable personas (74.24% consistency; hubs "
    "neutral-anchored). The ensemble trails validated transformers, so analyses should use the empirically "
    "selected scorer, pending confirmation on resolved reply graphs."
)

NEW_TABLE_ROWS = [
    ("xlm-twitter", "41.4", "44.9", "13.7"),
    ("bertweet", "11.3", "67.4", "21.2"),
]


def set_plain(para, text: str) -> None:
    """Replace paragraph text, preserving paragraph formatting (single plain run)."""
    p_el = para._element
    for r in para.runs:
        r.text = ""
    if para.runs:
        para.runs[0].text = text
    else:
        para.add_run(text)


def set_keywords(para) -> None:
    set_plain(para, "")
    para.runs[0].text = "Keywords: "
    para.runs[0].bold = True
    para.add_run("sentiment analysis; AI-to-AI interaction; thread dynamics; human validation")


def main() -> None:
    doc = Document(str(SRC))
    assert len(doc.paragraphs) == 16, f"unexpected paragraph count {len(doc.paragraphs)}"
    assert len(doc.tables) == 1 and len(doc.inline_shapes) == 1

    set_keywords(doc.paragraphs[5])
    set_plain(doc.paragraphs[7], INTRO)
    set_plain(doc.paragraphs[13], RESULTS)
    set_plain(doc.paragraphs[15], CONCLUSION)

    table = doc.tables[0]
    assert [c.text for c in table.rows[0].cells] == ["Scorer", "Negative", "Neutral", "Positive"]
    for row_vals in NEW_TABLE_ROWS:
        row = table.add_row()
        for cell, val in zip(row.cells, row_vals):
            cell.text = ""
            run = cell.paragraphs[0].add_run(val)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            rel.target_part._blob = FIG.read_bytes()
    for para in doc.paragraphs:
        if para.text.startswith("Figure 1."):
            set_plain(para, FIG_CAPTION)

    doc.save(str(DST))
    print(f"saved {DST}")
    print(f"intro chars: {len(INTRO)} (was 637)")
    print(f"results chars: {len(RESULTS)} (was 838)")
    print(f"conclusion chars: {len(CONCLUSION)} (was 322)")


if __name__ == "__main__":
    main()
