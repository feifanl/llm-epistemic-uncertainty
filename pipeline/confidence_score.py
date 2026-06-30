"""
Confidence scoring for steered/generated answers.

Replaces steer.py's old substring `hedge_count`, which was broken:
  - substring match: "may" fired on "Maya", "complex" on "complexity"
  - raw count, no length normalization -> longer answers look more hedged
  - hedge-only: couldn't tell neutral from assertive, so the +alpha (->known)
    direction only registered as hedge-absence
  - list hand-tuned to one model's phrasings -> won't transfer across families

Two scorers here:
  - lexical_score(): fast, free, offline. Word-boundary regex, BI-directional
    (hedge + assert lexicons), normalized per sentence. Use as a sanity signal
    and no-GPU fallback. Still keyword-brittle across model families -- the LLM
    judge (judge.py) is the primary cross-family metric.

The headline number is `net = assert_rate - hedge_rate`: higher = more
confident/assertive, lower = more hedged/uncertain. Steering toward "known"
(alpha>0) should raise it; toward "unknown" (alpha<0) should lower it.
"""
import re

# Multi-word phrases first so they match before their single-word substrings.
# Matched as whole words/phrases via \b boundaries (see _compile).
HEDGE_TERMS = [
    # explicit uncertainty
    "might", "may", "could", "possibly", "perhaps", "probably", "likely",
    "unlikely", "uncertain", "unclear", "not sure", "hard to say",
    "difficult to say", "depends", "depend on", "speculative", "debatable",
    # epistemic denial / "there is no settled answer" register
    "no single", "no one", "not a single", "no definitive", "no clear",
    "no consensus", "no scientific consensus", "no straightforward",
    "no evidence", "no easy", "no simple", "not straightforward",
    "is a myth", "a myth", "oversimplification", "complex", "complicated",
    "controversial", "multifaceted", "nuanced", "varies", "varying",
    "cannot predict", "can't predict", "cannot be certain",
    "impossible to predict", "no way to know", "difficult to predict",
    "i can't", "i cannot", "i'm not", "i am not",
]

ASSERT_TERMS = [
    "definitely", "certainly", "undoubtedly", "clearly", "obviously",
    "without a doubt", "of course", "indeed", "absolutely", "the answer is",
    "in fact", "simply", "is the", "are the", "always", "never", "must be",
    "the cause", "the reason", "yes,", "no,",  # direct verdicts
]

_SENT_SPLIT = re.compile(r"[.!?\n]+")


def _compile(terms):
    # sort longest-first so phrases win; \b on alnum edges, escape regex chars
    pats = sorted(terms, key=len, reverse=True)
    return [re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.I) for t in pats]


_HEDGE_RE = _compile(HEDGE_TERMS)
_ASSERT_RE = _compile(ASSERT_TERMS)


def _count(text, regexes):
    return sum(len(r.findall(text)) for r in regexes)


def _n_sentences(text):
    return max(1, len([s for s in _SENT_SPLIT.split(text) if s.strip()]))


def lexical_score(text):
    """Return dict of confidence signals for one answer string.

    hedge / assert      : raw whole-word phrase counts
    hedge_rate/assert_rate: per-sentence (length-normalized)
    net                 : assert_rate - hedge_rate  (HIGHER = more confident)
    """
    t = text or ""
    n_sent = _n_sentences(t)
    h = _count(t, _HEDGE_RE)
    a = _count(t, _ASSERT_RE)
    return {
        "hedge": h,
        "assert": a,
        "n_sent": n_sent,
        "hedge_rate": round(h / n_sent, 4),
        "assert_rate": round(a / n_sent, 4),
        "net": round((a - h) / n_sent, 4),
    }


if __name__ == "__main__":
    # quick smoke test on the two extremes from the Gemma run
    doubt = ('There is no single year for the French Revolution. '
             'The French Revolution is not a singular event, and the idea '
             'of a "revolution" is a myth. No consensus on the topic. '
             'No definitive answer.')
    sure = "The French Revolution began in 1789."
    for lbl, txt in [("doubt(-8)", doubt), ("sure(0)", sure)]:
        print(lbl, lexical_score(txt))
