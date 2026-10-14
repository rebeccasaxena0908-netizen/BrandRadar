"""Run the LLM stages by hand (the collector runs them every cycle when LLM_PROVIDER is set).

    python -m nlp.llm                          # translate, then labels and briefings (all brands)
    python -m nlp.llm --translate --brand Cosrx
    python -m nlp.llm --labels --briefings --force
    python -m nlp.llm --check                  # is the provider reachable?
    BRAND_RADAR_DEMO=1 python -m nlp.llm --labels --briefings   # AI labels and briefings for the demo data

Translated mentions still need the NLP pipeline (python -m nlp.pipeline) and topics; the collector does
that automatically right after translating.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from config import settings
from reports import quiet_streamlit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--translate", action="store_true", help="translate non-English mentions")
    parser.add_argument("--labels", action="store_true", help="LLM topic labels and descriptions")
    parser.add_argument("--briefings", action="store_true", help="written briefings (last 7 days + last week)")
    parser.add_argument("--brand", action="append", help="only this brand (repeatable)")
    parser.add_argument("--force", action="store_true", help="regenerate even when cached (and retry failed translations)")
    parser.add_argument("--redo", action="store_true", help="with --translate: translate already translated mentions again")
    parser.add_argument("--check", action="store_true", help="only check that the provider is reachable")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    quiet_streamlit()

    from db.session import get_session_factory
    from nlp.llm.provider import OllamaProvider, get_provider
    from nlp.pipeline import brand_ids

    provider = get_provider()
    print(f"LLM provider: {provider.name} ({provider.model})" + ("  · DEMO DATA" if settings.demo else ""))
    if isinstance(provider, OllamaProvider):
        ok, detail = provider.available()
        print(f"  {'✔' if ok else '✘'} {detail}")
        if not ok:
            return 1
    if args.check:
        return 0
    if provider.name == "none":
        print("  LLM_PROVIDER=none: nothing to generate (the dashboard uses the rule-based labels and summary).")
        return 0
    everything = not (args.translate or args.labels or args.briefings)
    factory = get_session_factory()
    with factory() as session:
        ids = brand_ids(session, args.brand)
    if args.translate or everything:
        from nlp.llm.translate import translate_pending

        out = translate_pending(factory, provider, ids, force=args.force, redo=args.redo)
        for brand in sorted(set(out.translated) | set(out.adverts) | set(out.failed)):
            print(f"  {brand:<10} translated {out.translated.get(brand, 0):>3} · adverts {out.adverts.get(brand, 0):>3}"
                  f" · failed {out.failed.get(brand, 0):>3}")
        if out.total:
            print("  -> run python -m nlp.pipeline, then python -m nlp.topics, to analyse the translations")
    if args.labels or everything:
        from nlp.llm.labels import refresh_labels

        print("  " + refresh_labels(factory, provider, ids, force=args.force).summary)
    if args.briefings or everything:
        from nlp.llm.briefing import refresh_briefings

        print("  " + refresh_briefings(factory, provider, ids, force=args.force).summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
