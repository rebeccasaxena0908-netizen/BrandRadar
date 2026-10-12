"""Check that the Brand Radar environment is ready.

Checks Python version, installed packages (against requirements.txt pins), .env,
the MySQL connection, that every table exists with utf8mb4, and prints row counts.
Exits with status 1 if any check fails.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

OK = "\033[32m✔\033[0m"
FAIL = "\033[31m✘\033[0m"
WARN = "\033[33m!\033[0m"

failures: list[str] = []


def report(passed: bool, label: str, detail: str = "", warn_only: bool = False) -> None:
    mark = OK if passed else (WARN if warn_only else FAIL)
    print(f"  {mark} {label}" + (f"  — {detail}" if detail else ""))
    if not passed and not warn_only:
        failures.append(label)


def section(title: str) -> None:
    print(f"\n{title}")


def check_python() -> None:
    section("Python")
    v = sys.version_info
    report(v[:2] == (3, 11), "Python 3.11", f"running {v.major}.{v.minor}.{v.micro}")
    in_venv = sys.prefix != sys.base_prefix
    report(in_venv, "Running inside a virtualenv", sys.prefix, warn_only=True)


def read_pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)$", line)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def check_packages() -> None:
    section("Packages (requirements.txt)")
    for name, pinned in read_pins().items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            report(False, name, "not installed")
            continue
        if installed == pinned:
            report(True, name, installed)
        else:
            report(False, name, f"installed {installed}, pinned {pinned}", warn_only=True)


def check_env() -> bool:
    section("Configuration (.env)")
    env_file = PROJECT_ROOT / ".env"
    report(env_file.exists(), ".env file present", str(env_file))
    try:
        from config import settings
    except Exception as exc:  # noqa: BLE001 — report any config error
        report(False, "config.py loads", repr(exc))
        return False
    report(True, "config.py loads", f"{settings.db_user}@{settings.db_host}:{settings.db_port}/{settings.db_name}")
    report(bool(settings.db_password), "DB_PASSWORD is set")
    report(
        settings.reddit_enabled,
        "Reddit credentials",
        "configured" if settings.reddit_enabled else "not set — Reddit source will be skipped",
        warn_only=True,
    )
    return env_file.exists()


def model_unique_keys() -> list[tuple[str, tuple[str, tuple[str, ...]]]]:
    from sqlalchemy import UniqueConstraint

    from db.models import Base

    return [
        (table.name, (str(con.name), tuple(c.name for c in con.columns)))
        for table in Base.metadata.sorted_tables
        for con in table.constraints
        if isinstance(con, UniqueConstraint)
    ]


def check_database() -> None:
    section("MySQL / MariaDB")
    try:
        from sqlalchemy import text

        from config import settings
        from db.models import TABLE_NAMES
        from db.session import get_engine
    except Exception as exc:  # noqa: BLE001
        report(False, "Import database modules", repr(exc))
        return

    try:
        with get_engine().connect() as conn:
            version = conn.execute(text("SELECT VERSION()")).scalar_one()
            report(True, "Connect to database", f"server {version}")

            charset = conn.execute(
                text(
                    "SELECT DEFAULT_CHARACTER_SET_NAME FROM information_schema.SCHEMATA "
                    "WHERE SCHEMA_NAME = :db"
                ),
                {"db": settings.db_name},
            ).scalar_one_or_none()
            report(charset == "utf8mb4", "Database charset utf8mb4", f"is {charset}")

            rows = conn.execute(
                text(
                    "SELECT TABLE_NAME, TABLE_COLLATION FROM information_schema.TABLES "
                    "WHERE TABLE_SCHEMA = :db"
                ),
                {"db": settings.db_name},
            ).all()
            existing = {name: collation for name, collation in rows}

            section("Tables")
            counts: dict[str, int] = {}
            for table in TABLE_NAMES:
                collation = existing.get(table)
                if collation is None:
                    report(False, table, "missing — run db/schema.sql")
                    continue
                utf8 = collation.startswith("utf8mb4")
                report(utf8, table, collation)
                counts[table] = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar_one()

            section("Unique keys (live database vs db/models.py)")
            live_uniques = conn.execute(
                text(
                    "SELECT TABLE_NAME, INDEX_NAME, GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) "
                    "FROM information_schema.STATISTICS "
                    "WHERE TABLE_SCHEMA = :db AND NON_UNIQUE = 0 AND INDEX_NAME <> 'PRIMARY' "
                    "GROUP BY TABLE_NAME, INDEX_NAME"
                ),
                {"db": settings.db_name},
            ).all()
            live = {(t, name, cols) for t, name, cols in live_uniques}
            for table, constraint in model_unique_keys():
                key = (table, constraint[0], ",".join(constraint[1]))
                report(key in live, f"{table}.{constraint[0]}", f"({', '.join(constraint[1])})")

            if counts:
                section("Row counts")
                width = max(len(t) for t in counts)
                for table, count in counts.items():
                    print(f"    {table:<{width}}  {count:>10,}")
    except Exception as exc:  # noqa: BLE001
        report(False, "Connect to database", f"{type(exc).__name__}: {exc}")


def main() -> int:
    print("Brand Radar setup check")
    check_python()
    check_packages()
    if check_env():
        check_database()

    print()
    if failures:
        print(f"{FAIL} {len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print(f"{OK} All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
