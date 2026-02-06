"""Metric catalog for label matching and unit expectations."""

from __future__ import annotations

from sqlalchemy.orm import selectinload

from qbr_intelligence.metrics.models import MetricAliasSpec, MetricCatalogEntry


def _aliases(*items: str) -> tuple[MetricAliasSpec, ...]:
    return tuple(MetricAliasSpec(alias=item) for item in items)


_DEFAULT_DISAMBIGUATION = {
    "unique_reach": ("unique", "deduplicated", "unduplicated"),
    "reach": (),
    "click_through_rate": ("click", "through"),
    "view_through_rate": ("view", "through"),
    "video_completion_rate": ("completion", "complete", "video"),
    "conversion_rate": ("conversion", "convert"),
    "cost_per_acquisition": ("acquisition", "acquired"),
    "cost_per_install": ("install",),
    "cost_per_engagement": ("engagement",),
    "cost_per_click": ("click",),
    "cost_per_mille": ("mille", "thousand"),
    "cost_per_view": ("view",),
    "install_lift": ("lift", "install"),
    "launch_lift": ("lift", "launch"),
    "roas": ("return", "spend"),
    "roi": ("return", "investment"),
    "share_of_voice": ("voice", "share"),
    "grps": ("rating", "gross"),
    "trps": ("rating", "target"),
    "household_reach": ("household", "hh"),
}


_DEF = [
        MetricCatalogEntry(
            metric_id="click_through_rate",
            name="Click Through Rate",
            aliases=_aliases("CTR", "click-through rate", "click through rate"),
            expected_unit="percent",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("click_through_rate", ()),
            priority=3,
        ),
        MetricCatalogEntry(
            metric_id="view_through_rate",
            name="View Through Rate",
            aliases=_aliases("VTR", "view-through rate", "view through rate"),
            expected_unit="percent",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("view_through_rate", ()),
            priority=3,
        ),
        MetricCatalogEntry(
            metric_id="video_completion_rate",
            name="Video Completion Rate",
            aliases=_aliases("VCR", "video completion rate", "completion rate"),
            expected_unit="percent",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("video_completion_rate", ()),
            priority=2,
        ),
        MetricCatalogEntry(
            metric_id="conversion_rate",
            name="Conversion Rate",
            aliases=_aliases("CVR", "conversion rate"),
            expected_unit="percent",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("conversion_rate", ()),
            priority=3,
        ),
        MetricCatalogEntry(
            metric_id="cost_per_acquisition",
            name="Cost per Acquisition",
            aliases=_aliases("CPA", "cost per acquisition", "cost per acquired user"),
            expected_unit="currency",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("cost_per_acquisition", ()),
            priority=3,
        ),
        MetricCatalogEntry(
            metric_id="cost_per_install",
            name="Cost per Install",
            aliases=_aliases("CPI", "cost per install"),
            expected_unit="currency",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("cost_per_install", ()),
            priority=3,
        ),
        MetricCatalogEntry(
            metric_id="cost_per_engagement",
            name="Cost per Engagement",
            aliases=_aliases("CPE", "cost per engagement"),
            expected_unit="currency",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("cost_per_engagement", ()),
            priority=3,
        ),
        MetricCatalogEntry(
            metric_id="cost_per_click",
            name="Cost per Click",
            aliases=_aliases("CPC", "cost per click"),
            expected_unit="currency",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("cost_per_click", ()),
            priority=3,
        ),
        MetricCatalogEntry(
            metric_id="cost_per_mille",
            name="Cost per Mille",
            aliases=_aliases("CPM", "cost per mille", "cost per thousand"),
            expected_unit="currency",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("cost_per_mille", ()),
            priority=3,
        ),
        MetricCatalogEntry(
            metric_id="cost_per_view",
            name="Cost per View",
            aliases=_aliases("CPV", "cost per view"),
            expected_unit="currency",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("cost_per_view", ()),
            priority=3,
        ),
        MetricCatalogEntry(
            metric_id="install_rate",
            name="Install Rate",
            aliases=_aliases("install rate"),
            expected_unit="percent",
            description=None,
            formula=None,
            priority=2,
        ),
        MetricCatalogEntry(
            metric_id="launch_rate",
            name="Launch Rate",
            aliases=_aliases("launch rate"),
            expected_unit="percent",
            description=None,
            formula=None,
            priority=2,
        ),
        MetricCatalogEntry(
            metric_id="install_lift",
            name="Install Lift",
            aliases=_aliases("install lift"),
            expected_unit="percent",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("install_lift", ()),
            priority=2,
        ),
        MetricCatalogEntry(
            metric_id="launch_lift",
            name="Launch Lift",
            aliases=_aliases("launch lift"),
            expected_unit="percent",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("launch_lift", ()),
            priority=2,
        ),
        MetricCatalogEntry(
            metric_id="roas",
            name="Return on Ad Spend",
            aliases=_aliases("ROAS", "return on ad spend"),
            expected_unit="ratio",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("roas", ()),
            priority=3,
        ),
        MetricCatalogEntry(
            metric_id="roi",
            name="Return on Investment",
            aliases=_aliases("ROI", "return on investment"),
            expected_unit="percent",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("roi", ()),
            priority=3,
        ),
        MetricCatalogEntry(
            metric_id="installs",
            name="Installs",
            aliases=_aliases("install", "installs"),
            expected_unit="count",
            description=None,
            formula=None,
            priority=2,
        ),
        MetricCatalogEntry(
            metric_id="launches",
            name="Launches",
            aliases=_aliases("launches", "launch"),
            expected_unit="count",
            description=None,
            formula=None,
            priority=2,
        ),
        MetricCatalogEntry(
            metric_id="unique_reach",
            name="Unique Reach",
            aliases=_aliases(
                "unique reach",
                "deduplicated reach",
                "de-duplicated reach",
                "unduplicated reach",
            ),
            expected_unit="count",
            description=None,
            formula=None,
            disambiguation=_DEFAULT_DISAMBIGUATION.get("unique_reach", ()),
            priority=3,
        ),
    MetricCatalogEntry(
        metric_id="monthly_active_users",
        name="Monthly Active Users",
        aliases=_aliases("MAU", "monthly active users"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="daily_active_users",
        name="Daily Active Users",
        aliases=_aliases("DAU", "daily active users"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="reach",
        name="Reach",
        aliases=_aliases("reach"),
        expected_unit="count",
        priority=1,
    ),
    MetricCatalogEntry(
        metric_id="household_reach",
        name="Household Reach",
        aliases=_aliases("household reach", "hh reach"),
        expected_unit="count",
        disambiguation=_DEFAULT_DISAMBIGUATION.get("household_reach", ()),
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="share_of_voice",
        name="Share of Voice",
        aliases=_aliases("share of voice", "SOV"),
        expected_unit="percent",
        disambiguation=_DEFAULT_DISAMBIGUATION.get("share_of_voice", ()),
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="grps",
        name="GRPs",
        aliases=_aliases("GRP", "GRPs", "gross rating point", "gross rating points"),
        expected_unit="count",
        disambiguation=_DEFAULT_DISAMBIGUATION.get("grps", ()),
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="trps",
        name="TRPs",
        aliases=_aliases("TRP", "TRPs", "target rating point", "target rating points"),
        expected_unit="count",
        disambiguation=_DEFAULT_DISAMBIGUATION.get("trps", ()),
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="impressions",
        name="Impressions",
        aliases=_aliases("impression", "impressions"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="clicks",
        name="Clicks",
        aliases=_aliases("click", "clicks"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="views",
        name="Views",
        aliases=_aliases("view", "views", "video views"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="completes",
        name="Completes",
        aliases=_aliases("complete", "completes", "video completion", "video completions"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="spend",
        name="Spend",
        aliases=_aliases("spend", "spending", "investment"),
        expected_unit="currency",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="csrb_spend",
        name="Content Store Roadblock Spend",
        aliases=_aliases("CSRB", "content store roadblock", "content store rb"),
        expected_unit="currency",
        disambiguation=_DEFAULT_DISAMBIGUATION.get("csrb_spend", ()),
        priority=1,
    ),
    MetricCatalogEntry(
        metric_id="acquisitions",
        name="Acquisitions",
        aliases=_aliases("acquisitions", "acquired users", "acquired user"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="qualified_acquisitions",
        name="Qualified Acquisitions",
        aliases=_aliases("qualified acquisitions", "qualified acquired users", "qualified acquired user"),
        expected_unit="count",
        disambiguation=("qualified",),
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="conversions",
        name="Conversions",
        aliases=_aliases("conversions", "converted users", "converted user"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="purchases",
        name="Purchases",
        aliases=_aliases("purchases", "purchase", "transactions", "transaction"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="sign_ups",
        name="Sign-ups",
        aliases=_aliases("sign up", "sign-up", "signups", "sign ups", "sign-in", "sign in"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="leads",
        name="Leads",
        aliases=_aliases("lead", "leads"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="engagements",
        name="Engagements",
        aliases=_aliases("engagement", "engagements", "engaged users", "engaged user"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="sessions",
        name="Sessions",
        aliases=_aliases("session", "sessions", "session count"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="frequency",
        name="Frequency",
        aliases=_aliases("frequency"),
        expected_unit="count",
        priority=1,
    ),
    MetricCatalogEntry(
        metric_id="video_starts",
        name="Video Starts",
        aliases=_aliases("video starts", "video start"),
        expected_unit="count",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="fill_rate",
        name="Fill Rate",
        aliases=_aliases("fill rate"),
        expected_unit="percent",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="win_rate",
        name="Win Rate",
        aliases=_aliases("win rate"),
        expected_unit="percent",
        priority=2,
    ),
    MetricCatalogEntry(
        metric_id="viewability",
        name="Viewability",
        aliases=_aliases("viewability"),
        expected_unit="percent",
        priority=2,
    ),
]


def build_metric_catalog() -> list[MetricCatalogEntry]:
    return list(_DEF)


def build_metric_catalog_from_db(session) -> list[MetricCatalogEntry]:
    from qbr_intelligence.db.models import MetricCatalog

    rows = (
        session.query(MetricCatalog)
        .options(selectinload(MetricCatalog.aliases))
        .all()
    )
    entries: list[MetricCatalogEntry] = []
    for row in rows:
        aliases = []
        for alias in row.aliases or []:
            aliases.append(
                MetricAliasSpec(
                    alias=alias.alias,
                    pattern=alias.pattern,
                    priority=alias.priority,
                    unit_override=alias.unit_override,
                )
            )
        if not aliases:
            aliases = [MetricAliasSpec(alias=row.name)]
        metric_id = row.slug or row.name
        disambiguation = _DEFAULT_DISAMBIGUATION.get(metric_id, ())
        entries.append(
            MetricCatalogEntry(
                metric_id=metric_id,
                name=row.name,
                aliases=tuple(aliases),
                expected_unit=row.default_unit or "unknown",
                description=row.description,
                formula=row.formula,
                disambiguation=disambiguation,
                priority=0,
            )
        )
    return entries


def catalog_by_id(entries: list[MetricCatalogEntry] | None = None) -> dict[str, MetricCatalogEntry]:
    source = entries if entries is not None else _DEF
    return {entry.metric_id: entry for entry in source}


def catalog_by_name(entries: list[MetricCatalogEntry] | None = None) -> dict[str, MetricCatalogEntry]:
    source = entries if entries is not None else _DEF
    return {entry.name: entry for entry in source}
