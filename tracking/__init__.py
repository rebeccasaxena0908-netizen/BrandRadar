"""Track any brand from the dashboard (Step 11).

    store.py    brand_requests: statuses, validation, the dashboard's writes (only to this table)
    apps.py     App Store / Google Play id lookup by name
    setup.py    research -> a proposed group (profile, competitors, matching rules), checks, brands.yaml
    worker.py   the collector's request job: research, dry-run preview, then fetch and analyse the new group
"""
