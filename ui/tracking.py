"""The dashboard's "Track a new brand" card (Step 11).

The one place the dashboard writes, and only to brand_requests (tracking/store.py): a new request, the
review decision (approve, edit and re-check, cancel) and group removals. The collector does the research,
the dry run, brands.yaml and the pipeline. The request list refreshes every 10 s in a fragment; the review
panel sits outside it, so a form being edited is never redrawn under the person's cursor (the list asks
for one full rerun when a request becomes ready for review).
"""

from __future__ import annotations

import html
import time
from typing import Any

import streamlit as st
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from db.models import Brand, BrandRequest
from nlp.profiles import load_profiles
from tracking import setup as setup_mod
from tracking import store
from tracking.store import (
    CANCELLED, FAILED, NOT_ENOUGH, READY, REMOVE, REMOVED, REVIEW, RequestError,
)
from ui import data
from ui.components import glass_card, section_title

STATUS_CLASS = {READY: "ok", NOT_ENOUGH: "warn", FAILED: "bad", CANCELLED: "off", REMOVED: "off", REVIEW: "review"}


def session() -> Session:
    return Session(data.get_engine(), expire_on_commit=False)


def esc(value: Any) -> str:
    return html.escape(str(value))


def flash(kind: str, message: str) -> None:
    st.session_state["track-flash"] = (kind, message, time.monotonic())


def show_flash(seconds: float = 8.0) -> None:
    """The last action's message, kept for a few seconds: an action is often followed by a second rerun
    (the review panel closes), which would swallow a show-once message."""
    msg = st.session_state.get("track-flash")
    if msg and time.monotonic() - msg[2] < seconds:
        st.markdown(f'<div class="br-track-msg br-track-msg-{msg[0]}">{esc(msg[1])}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- card


def track_card() -> None:
    with glass_card("track"):
        section_title("Track a new brand", "Any brand, any industry: the collector researches its market, competitors "
                      "and how people write it, you review the setup, then it starts tracking")
        if settings.demo:
            st.markdown('<div class="br-note">Not available on demo data: the collector only works on the real '
                        'database.</div>', unsafe_allow_html=True)
            return
        with st.form("track-form", clear_on_submit=True, border=False):
            name_col, button_col = st.columns([4, 1], vertical_alignment="bottom")
            name = name_col.text_input("Brand", placeholder="e.g. McDonald's, Nivea, Tesla, Zomato",
                                       label_visibility="collapsed", max_chars=60, key="track-name")
            submitted = button_col.form_submit_button("Track", type="primary", use_container_width=True)
        if submitted:
            try:
                with session() as s:
                    req = store.create_request(s, name)
                flash("ok", f"{req.name} is queued. The collector picks it up within a minute.")
            except RequestError as exc:
                flash("bad", str(exc))
        show_flash()
        request_list()
        for rid in st.session_state.get("track-review-ids", []):
            review_panel(rid)
        removal_section()


@st.fragment(run_every="10s")
def request_list() -> None:
    with session() as s:
        rows = store.recent(s)
    review_ids = [r.id for r in rows if r.kind != REMOVE and r.status == REVIEW]
    if review_ids != st.session_state.get("track-review-ids"):
        first = "track-review-ids" not in st.session_state
        st.session_state["track-review-ids"] = review_ids
        if not first:
            st.rerun(scope="app")  # show (or hide) the review panels, which live outside this fragment
    if not rows:
        st.markdown('<div class="br-note">No brands requested yet.</div>', unsafe_allow_html=True)
        return
    items = "".join(request_row(r) for r in rows)
    st.markdown(f'<ul class="br-track-list">{items}</ul>', unsafe_allow_html=True)


def request_row(r: BrandRequest) -> str:
    label = store.LABELS.get(r.status, r.status)
    if r.kind == REMOVE:
        label = {"queued": "Removal queued", "removing": "Removing", "removed": "Removed"}.get(r.status, label)
    cls = STATUS_CLASS.get(r.status, "busy")
    what = f"Remove {r.group_name}" if r.kind == REMOVE else r.name
    group = f' <span class="br-track-group">· {esc(r.group_name)}</span>' if r.group_name and r.kind != REMOVE else ""
    detail = r.error if r.status == FAILED and r.error else (r.detail or "")
    extra = result_html(r) if r.kind != REMOVE and r.status in (READY, NOT_ENOUGH) else ""
    return (f'<li class="br-track-item"><div class="br-track-top"><b>{esc(what)}</b>{group}'
            f'<span class="br-track-pill br-track-{cls}">{esc(label)}</span>'
            f'<span class="br-track-age">{esc(store.age(r.updated_at))}</span></div>'
            f'<div class="br-track-detail">{esc(detail)}</div>{extra}</li>')


def result_html(r: BrandRequest) -> str:
    res = r.result or {}
    mentions: dict[str, dict[str, int]] = res.get("mentions") or {}
    if not mentions:
        return ""
    sources = sorted({s for per in mentions.values() for s in per})
    head = "".join(f"<th>{esc(setup_mod.data_name(s))}</th>" for s in sources)
    body = "".join(
        f"<tr><td>{esc(b)}</td>" + "".join(f"<td>{per.get(s, 0)}</td>" for s in sources)
        + f"<td><b>{sum(per.values())}</b></td><td>{(res.get('analysed') or {}).get(b, 0)}</td></tr>"
        for b, per in mentions.items())
    topics = ", ".join(f"{t['label']} ({t['mentions']})" for t in res.get("topics") or [])
    secs = res.get("seconds") or {}
    took = f" · took {sum(secs.values()) / 60:.0f} min" if secs else ""
    hint = ""
    if r.status == NOT_ENOUGH:
        hint = ('<div class="br-track-hint">Few public posts so far. The collector keeps adding mentions every 6 hours; '
                'exported reviews can be imported with <code>python fetch.py --csv file.csv --brand '
                f'"{esc(next(iter(mentions)))}"</code>.</div>')
    errors = "".join(f"<li>{esc(e)}</li>" for e in (res.get("source_errors") or [])[:6])
    errors = f'<ul class="br-track-errors">{errors}</ul>' if errors else ""
    return (f'<table class="br-track-table"><tr><th>Brand</th>{head}<th>Total</th><th>Analysed</th></tr>{body}</table>'
            f'<div class="br-track-detail">Top complaint topics: {esc(topics) if topics else "none yet"}{esc(took)}</div>'
            f'{hint}{errors}')


# ---------------------------------------------------------------- review


def review_panel(rid: int) -> None:
    with session() as s:
        req = s.get(BrandRequest, rid)
        tracked = [n for (n,) in s.execute(select(Brand.name))]
        groups = [g for (g,) in s.execute(select(Brand.brand_group).distinct())]
    if req is None or req.status != REVIEW or not req.setup:
        return
    setup: dict[str, Any] = dict(req.setup)
    preview: dict[str, Any] = dict(req.preview or {})
    with st.container(key=f"track-review-{rid}"):
        st.markdown(f'<div class="br-track-review-head">Review: <b>{esc(req.name)}</b></div>', unsafe_allow_html=True)
        if req.error:
            st.markdown(f'<div class="br-track-msg br-track-msg-bad">{esc(req.error)}</div>', unsafe_allow_html=True)
        st.markdown(setup_html(setup) + preview_html(preview, setup), unsafe_allow_html=True)
        problems = setup_mod.problems(setup, tracked, groups)
        editing = st.session_state.get(f"track-edit-{rid}", bool(setup.get("needs_input")))
        if editing:
            edit_form(rid, setup, tracked, groups)
            if st.button("Cancel this request", key=f"track-cancel-{rid}", icon=":material/close:"):
                with session() as s:
                    store.cancel(s, rid)
                flash("ok", f"{req.name} cancelled. Nothing was added.")
                st.rerun()
            return
        if problems:
            st.markdown('<div class="br-track-msg br-track-msg-bad">' + esc(" ".join(problems)) + "</div>",
                        unsafe_allow_html=True)
        a, e, c, _ = st.columns([1.2, 1, 1, 3])
        if a.button("Approve", key=f"track-approve-{rid}", type="primary", disabled=bool(problems),
                    icon=":material/check:"):
            with session() as s:
                ok = store.approve(s, rid)
            flash("ok" if ok else "bad", f"{req.name} approved: fetching starts within a minute." if ok
                  else "That request changed meanwhile; nothing was done.")
            st.rerun()
        if e.button("Edit", key=f"track-edit-btn-{rid}", icon=":material/edit:"):
            st.session_state[f"track-edit-{rid}"] = True
            st.rerun()
        if c.button("Cancel", key=f"track-cancel-{rid}", icon=":material/close:"):
            with session() as s:
                store.cancel(s, rid)
            flash("ok", f"{req.name} cancelled. Nothing was added.")
            st.rerun()


def chips(items: list[str] | None, empty: str = "–") -> str:
    if not items:
        return f'<span class="br-track-none">{esc(empty)}</span>'
    return "".join(f'<span class="br-track-chip">{esc(i)}</span>' for i in items)


def setup_html(setup: dict[str, Any]) -> str:
    draft = setup.get("profile_draft")
    if draft and draft.get("name") == setup.get("profile"):
        cats = ", ".join(draft["raw"].get("complaints") or [])
        profile = f"{setup['profile']} <em>(new, drafted for “{esc(setup.get('industry'))}”: {esc(cats)})</em>"
    else:
        profile = esc(setup.get("profile") or "– pick one –")
    rows = []
    for b in setup.get("brands") or []:
        apps = " · ".join(x for x in (b.get("app_store_title") and f"App Store: {b['app_store_title']}",
                                      b.get("play_store_title") and f"Google Play: {b['play_store_title']}") if x)
        rows.append(
            f'<tr><td><b>{esc(b["name"])}</b>{" (own)" if b.get("is_own") else ""}'
            f'<div class="br-track-small">{"LLM" if b.get("rules") == "llm" else "rule-based"}'
            f'{" · also means: " + esc("; ".join(b["other_meanings"])) if b.get("other_meanings") else ""}</div></td>'
            f'<td>{chips(b.get("keywords"))}</td><td>{chips(b.get("exclude"))}</td><td>{chips(b.get("exclude_if"))}</td>'
            f'<td>{chips(b.get("require_context"))}</td><td>{chips(b.get("youtube_queries"))}'
            f'<div class="br-track-small">{esc(apps)}</div></td></tr>')
    notes = "".join(f"<li>{esc(n)}</li>" for n in setup.get("notes") or [])
    return (f'<div class="br-track-setup"><div><span class="br-label">Industry</span> {esc(setup.get("industry") or "–")}'
            f' &nbsp; <span class="br-label">Profile</span> {profile} &nbsp; <span class="br-label">New group</span> '
            f'{esc(setup.get("group") or "–")}</div>'
            '<div class="br-track-scroll"><table class="br-track-table"><tr><th>Brand</th><th>Keywords</th>'
            '<th>Cut out (exclude)</th><th>Drop if (exclude_if)</th><th>Must mention (context)</th>'
            f'<th>YouTube · apps</th></tr>{"".join(rows)}</table></div>'
            f'{f"<ul class=br-track-notes>{notes}</ul>" if notes else ""}</div>')


def preview_html(preview: dict[str, Any], setup: dict[str, Any]) -> str:
    if not preview:
        return ""
    counts = []
    for brand, per in (preview.get("brands") or {}).items():
        parts = ", ".join(f"{setup_mod.data_name(s)} {c['matched']} matched / {c['rejected']} rejected" for s, c in per.items())
        counts.append(f"<li><b>{esc(brand)}</b>: {esc(parts) or 'no source answered'}</li>")
    matched = "".join(f'<li><span class="br-track-src">{esc(setup_mod.data_name(s))}</span> {esc(t)}</li>'
                      for s, t in preview.get("matched") or [])
    rejected = "".join(f'<li><span class="br-track-src">{esc(setup_mod.data_name(s))} · {esc(r)}</span> {esc(t)}</li>'
                       for s, r, t in preview.get("rejected") or [])
    notes = "".join(f"<li>{esc(n)}</li>" for n in (preview.get("notes") or [])[:8])
    own = next((b["name"] for b in setup.get("brands") or [] if b.get("is_own")), "")
    took = preview.get("seconds", 0) + preview.get("research_seconds", 0)
    return (f'<div class="br-track-preview"><div class="br-label">Dry run, last 30 days, nothing stored '
            f'({took:.0f} s)</div><ul class="br-track-counts">{"".join(counts)}</ul>'
            f'<div class="br-track-samples"><div><div class="br-label">Matched for {esc(own)} (sample)</div>'
            f'<ul>{matched or "<li>none</li>"}</ul></div><div><div class="br-label">Rejected (sample)</div>'
            f'<ul>{rejected or "<li>none</li>"}</ul></div></div>'
            f'{f"<ul class=br-track-notes>{notes}</ul>" if notes else ""}</div>')


def split(text: str) -> list[str]:
    return list(dict.fromkeys(p.strip() for p in (text or "").split(",") if p.strip()))


def edit_form(rid: int, setup: dict[str, Any], tracked: list[str], groups: list[str]) -> None:
    profiles = list(load_profiles())
    draft = setup.get("profile_draft")
    if draft and draft.get("name") not in profiles:
        profiles.append(draft["name"])
    with st.form(f"track-edit-form-{rid}", border=False):
        if setup.get("needs_input"):
            st.markdown('<div class="br-track-msg br-track-msg-warn">Pick the industry profile and type the '
                        'competitors, then re-check. Their matching rules are filled in during the re-check.</div>',
                        unsafe_allow_html=True)
        g, p = st.columns(2)
        group = g.text_input("Group name", value=setup.get("group") or "", key=f"te-group-{rid}")
        current = setup.get("profile")
        profile = p.selectbox("Industry profile", profiles, index=profiles.index(current) if current in profiles else None,
                              placeholder="Pick one", key=f"te-profile-{rid}")
        edited: list[dict[str, Any]] = []
        for i, b in enumerate(setup.get("brands") or []):
            st.markdown(f'<div class="br-track-edit-brand">{esc(b["name"])}{" (own brand)" if b.get("is_own") else ""}</div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            row = dict(b)
            row["keywords"] = split(c1.text_input("Keywords", ", ".join(b.get("keywords") or []), key=f"te-kw-{rid}-{i}"))
            row["exclude"] = split(c2.text_input("Cut out (exclude)", ", ".join(b.get("exclude") or []), key=f"te-ex-{rid}-{i}"))
            row["exclude_if"] = split(c3.text_input("Drop if (exclude_if)", ", ".join(b.get("exclude_if") or []), key=f"te-ei-{rid}-{i}"))
            c4, c5, c6 = st.columns(3)
            row["require_context"] = split(c4.text_input("Must mention one of", ", ".join(b.get("require_context") or []), key=f"te-rc-{rid}-{i}"))
            row["youtube_queries"] = split(c5.text_input("YouTube queries", ", ".join(b.get("youtube_queries") or []), key=f"te-yt-{rid}-{i}"))
            keep_apps = c6.checkbox("Use the app reviews found", value=bool(b.get("app_store_id") or b.get("play_store_id")),
                                    key=f"te-apps-{rid}-{i}", disabled=not (b.get("app_store_id") or b.get("play_store_id")))
            if not keep_apps:
                row.update(app_store_id=None, app_store_title=None, play_store_id=None, play_store_title=None)
            drop = False if b.get("is_own") else c6.checkbox("Leave this competitor out", key=f"te-drop-{rid}-{i}")
            if not drop:
                edited.append(row)
        added = split(st.text_input("Add competitors (comma-separated)", key=f"te-add-{rid}",
                                    placeholder="e.g. Burger King, KFC"))
        a, r, c, _ = st.columns([1.6, 1.4, 1, 2])
        approve = a.form_submit_button("Approve with these settings", type="primary")
        recheck = r.form_submit_button("Save and re-check")
        close = c.form_submit_button("Close")
    if close:
        st.session_state[f"track-edit-{rid}"] = False
        st.rerun()
    if not (approve or recheck):
        return
    names = {setup_mod.key(b["name"]) for b in edited}
    for name in added:
        if setup_mod.key(name) not in names:
            edited.append({"name": " ".join(name.split()), "is_own": False, "keywords": []})
            names.add(setup_mod.key(name))
    new = {**setup, "group": " ".join(group.split()), "profile": profile, "brands": edited, "needs_input": False}
    problems = setup_mod.problems(new, tracked, groups)
    needs_research = any(not b.get("keywords") for b in edited)
    with session() as s:
        if approve and not needs_research and not problems:
            ok = store.approve(s, rid, new)
            msg = "Approved: fetching starts within a minute."
        else:
            ok = store.recheck(s, rid, new)
            msg = ("New competitors need their matching rules first: re-checking." if approve and needs_research
                   else "Saved: re-checking with the dry run." if not problems
                   else "Saved and re-checking. Still to fix before approving: " + " ".join(problems))
    st.session_state[f"track-edit-{rid}"] = False
    flash("ok" if ok else "bad", msg if ok else "That request changed meanwhile; nothing was saved.")
    st.rerun()


# ---------------------------------------------------------------- removal


def removal_section() -> None:
    with session() as s:
        groups = store.removable_groups(s)
    if not groups:
        return
    with st.expander("Remove a group added here"):
        st.markdown('<div class="br-note">Deletes the group\'s brands, mentions, analysis and topics. Groups set '
                    'up in brands.yaml by hand can only be removed there.</div>', unsafe_allow_html=True)
        g, t, b = st.columns([1.3, 1.5, 1.2], vertical_alignment="bottom")
        group = g.selectbox("Group to remove", groups, key="track-remove-group")
        typed = t.text_input("Type the group name to confirm", key="track-remove-confirm")
        if b.button("Delete group and its data", key="track-remove-go", disabled=typed.strip() != group,
                    icon=":material/delete:"):
            try:
                with session() as s:
                    store.request_removal(s, group, typed)
                flash("ok", f"{group} will be removed within a minute.")
            except RequestError as exc:
                flash("bad", str(exc))
            st.rerun()
