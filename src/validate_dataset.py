"""생성된 7개 CSV의 무결성과 분석 분포를 검증하고 JSON 보고서를 작성한다."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TABLES = [
    "users", "listings", "product_views", "favorites", "chatrooms",
    "transactions", "price_history",
]

PRIMARY_KEYS = {
    "users": "user_id",
    "listings": "listing_id",
    "product_views": "view_id",
    "favorites": "favorite_id",
    "chatrooms": "chatroom_id",
    "transactions": "transaction_id",
    "price_history": "price_history_id",
}

NOT_NULL_COLUMNS = {
    "users": ["user_id", "signup_at", "region_id", "manner_score", "is_verified", "last_active_at"],
    "listings": ["listing_id", "seller_id", "category", "title", "description", "photo_count", "price", "is_negotiable", "region_id", "created_at", "status"],
    "product_views": ["view_id", "user_id", "listing_id", "viewed_at", "source"],
    "favorites": ["favorite_id", "user_id", "listing_id", "added_view_id", "favorited_at"],
    "chatrooms": ["chatroom_id", "listing_id", "buyer_id", "seller_id", "created_at", "message_count", "chat_status"],
    "transactions": ["transaction_id", "listing_id", "chatroom_id", "buyer_id", "seller_id", "transaction_status", "started_at"],
    "price_history": ["price_history_id", "listing_id", "price", "record_type", "recorded_at"],
}

DATETIME_COLUMNS = {
    "users": ["signup_at", "last_active_at"],
    "listings": ["created_at"],
    "product_views": ["viewed_at"],
    "favorites": ["favorited_at", "removed_at"],
    "chatrooms": ["created_at", "first_message_at", "seller_response_at"],
    "transactions": ["started_at", "completed_at", "ended_at"],
    "price_history": ["recorded_at"],
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"JSON으로 변환할 수 없는 값: {type(value).__name__}")


def load_generated_tables(project_root: Path) -> dict[str, pd.DataFrame]:
    """generated/csv의 7개 테이블을 읽고 DATETIME 컬럼을 파싱한다."""

    tables: dict[str, pd.DataFrame] = {}
    for table in TABLES:
        path = project_root / "generated/csv" / f"{table}.csv"
        if not path.exists():
            raise FileNotFoundError(f"필수 CSV가 없습니다: {path}")
        frame = pd.read_csv(path)
        for column in DATETIME_COLUMNS[table]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        tables[table] = frame
    return tables


def _issue(
    collection: list[dict[str, Any]],
    code: str,
    message: str,
    count: int,
    sample: list[Any] | None = None,
) -> None:
    if count:
        collection.append(
            {
                "code": code,
                "message": message,
                "count": int(count),
                "sample": sample[:10] if sample else [],
            }
        )


def _count_mask(mask: pd.Series) -> int:
    return int(mask.fillna(False).sum())


def _distribution(series: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = series.value_counts(dropna=False)
    total = max(1, int(counts.sum()))
    return {
        str(key): {"count": int(value), "ratio": float(value / total)}
        for key, value in counts.items()
    }


def _build_latest_activity(tables: dict[str, pd.DataFrame]) -> dict[int, pd.Timestamp]:
    users = tables["users"]
    latest = {int(row.user_id): row.signup_at for row in users.itertuples()}

    def update(user_id: int, timestamp: Any) -> None:
        if pd.notna(timestamp) and timestamp > latest[int(user_id)]:
            latest[int(user_id)] = timestamp

    for row in tables["listings"].itertuples():
        update(row.seller_id, row.created_at)
    for row in tables["product_views"].itertuples():
        update(row.user_id, row.viewed_at)
    for row in tables["favorites"].itertuples():
        update(row.user_id, row.favorited_at)
        update(row.user_id, row.removed_at)
    for row in tables["chatrooms"].itertuples():
        update(row.buyer_id, row.created_at)
        update(row.buyer_id, row.first_message_at)
        update(row.seller_id, row.seller_response_at)
    for row in tables["transactions"].itertuples():
        final_time = row.completed_at if pd.notna(row.completed_at) else row.ended_at
        if pd.isna(final_time):
            final_time = row.started_at
        update(row.buyer_id, final_time)
        update(row.seller_id, final_time)
    return latest


def _daily_activity(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    timestamp_map = {
        "product_views": "viewed_at",
        "favorites": "favorited_at",
        "chatrooms": "created_at",
        "transactions": "started_at",
    }
    combined: Counter[str] = Counter()
    result: dict[str, dict[str, int]] = {}
    for table, column in timestamp_map.items():
        counts = tables[table][column].dt.strftime("%Y-%m-%d").value_counts().sort_index()
        result[table] = {str(key): int(value) for key, value in counts.items()}
        combined.update(result[table])
    result["combined"] = dict(sorted(combined.items()))
    return result


def _outside_interest_ratio(
    project_root: Path,
    tables: dict[str, pd.DataFrame],
) -> float:
    profile_path = project_root / "generated/persona_profiles.json"
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))["profiles"]
    interests = {
        int(item["user_id"]): set(item["primary_interests"] + item["secondary_interests"])
        for item in profiles
    }
    listing_categories = tables["listings"].set_index("listing_id")["category"].to_dict()
    outside = 0
    for row in tables["product_views"].itertuples():
        outside += listing_categories[int(row.listing_id)] not in interests[int(row.user_id)]
    return float(outside / max(1, len(tables["product_views"])))


def _price_distribution(listings: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for category, group in listings.groupby("category"):
        prices = group["price"].astype(float)
        result[str(category)] = {
            "count": int(len(prices)),
            "min": int(prices.min()),
            "p25": float(prices.quantile(0.25)),
            "median": float(prices.median()),
            "p75": float(prices.quantile(0.75)),
            "max": int(prices.max()),
            "mean": float(prices.mean()),
        }
    return result


def validate_dataset(
    config: dict[str, Any],
    project_root: Path,
    simulation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """치명적 무결성 오류와 비치명적 분포 경고를 구분해 보고한다."""

    tables = load_generated_tables(project_root)
    fatal_errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    row_counts = {table: int(len(frame)) for table, frame in tables.items()}

    for table, frame in tables.items():
        pk = PRIMARY_KEYS[table]
        duplicates = frame[pk].duplicated(keep=False)
        _issue(fatal_errors, f"{table}_pk_duplicate", f"{table}.{pk} 중복", _count_mask(duplicates), frame.loc[duplicates, pk].tolist())
        for column in NOT_NULL_COLUMNS[table]:
            missing = frame[column].isna()
            if frame[column].dtype == object:
                missing = missing | frame[column].astype(str).str.strip().eq("")
            _issue(fatal_errors, f"{table}_{column}_not_null", f"{table}.{column} NOT NULL 위반", _count_mask(missing), frame.loc[missing, pk].tolist())

    users = tables["users"]
    listings = tables["listings"]
    views = tables["product_views"]
    favorites = tables["favorites"]
    chats = tables["chatrooms"]
    transactions = tables["transactions"]
    prices = tables["price_history"]

    fk_specs = [
        ("listings", "seller_id", "users", "user_id"),
        ("product_views", "user_id", "users", "user_id"),
        ("product_views", "listing_id", "listings", "listing_id"),
        ("favorites", "user_id", "users", "user_id"),
        ("favorites", "listing_id", "listings", "listing_id"),
        ("favorites", "added_view_id", "product_views", "view_id"),
        ("favorites", "removed_view_id", "product_views", "view_id"),
        ("chatrooms", "listing_id", "listings", "listing_id"),
        ("chatrooms", "buyer_id", "users", "user_id"),
        ("chatrooms", "seller_id", "users", "user_id"),
        ("transactions", "listing_id", "listings", "listing_id"),
        ("transactions", "chatroom_id", "chatrooms", "chatroom_id"),
        ("transactions", "buyer_id", "users", "user_id"),
        ("transactions", "seller_id", "users", "user_id"),
        ("price_history", "listing_id", "listings", "listing_id"),
    ]
    for child_table, child_col, parent_table, parent_col in fk_specs:
        child = tables[child_table]
        values = child[child_col].dropna()
        invalid_values = values[~values.isin(tables[parent_table][parent_col])]
        _issue(
            fatal_errors,
            f"{child_table}_{child_col}_fk",
            f"{child_table}.{child_col} → {parent_table}.{parent_col} FK 불일치",
            int(len(invalid_values)), invalid_values.tolist(),
        )

    _issue(fatal_errors, "users_manner_score_check", "manner_score 범위 위반", _count_mask(~users["manner_score"].between(0, 100)))
    _issue(fatal_errors, "users_boolean_check", "is_verified BOOLEAN 위반", _count_mask(~users["is_verified"].isin([0, 1])))
    _issue(fatal_errors, "users_time_check", "last_active_at이 signup_at보다 빠름", _count_mask(users["last_active_at"] < users["signup_at"]))
    _issue(fatal_errors, "listings_photo_check", "photo_count 음수", _count_mask(listings["photo_count"] < 0))
    _issue(fatal_errors, "listings_price_check", "listings.price 음수", _count_mask(listings["price"] < 0))
    _issue(fatal_errors, "listings_boolean_check", "is_negotiable BOOLEAN 위반", _count_mask(~listings["is_negotiable"].isin([0, 1])))
    _issue(fatal_errors, "listings_status_check", "listings.status 허용값 위반", _count_mask(~listings["status"].isin(["판매중", "예약중", "판매완료", "삭제"])))
    _issue(fatal_errors, "views_source_check", "product_views.source 허용값 위반", _count_mask(~views["source"].isin(["search", "recommendation", "category", "favorite_list", "seller_profile", "other"])))
    _issue(fatal_errors, "chat_buyer_seller_check", "채팅 구매자와 판매자가 같음", _count_mask(chats["buyer_id"] == chats["seller_id"]))
    _issue(fatal_errors, "chat_message_count_check", "message_count 음수", _count_mask(chats["message_count"] < 0))
    _issue(fatal_errors, "chat_status_check", "chat_status 허용값 위반", _count_mask(~chats["chat_status"].isin(["진행중", "종료", "차단"])))
    _issue(fatal_errors, "transaction_buyer_seller_check", "거래 구매자와 판매자가 같음", _count_mask(transactions["buyer_id"] == transactions["seller_id"]))
    _issue(fatal_errors, "transaction_status_check", "transaction_status 허용값 위반", _count_mask(~transactions["transaction_status"].isin(["거래중", "거래완료", "거래종료"])))
    _issue(fatal_errors, "price_history_price_check", "price_history.price 음수", _count_mask(prices["price"] < 0))
    _issue(fatal_errors, "price_history_type_check", "record_type 허용값 위반", _count_mask(~prices["record_type"].isin(["initial", "changed"])))

    user_signup = users.set_index("user_id")["signup_at"]
    user_verified = users.set_index("user_id")["is_verified"]
    listing_created = listings.set_index("listing_id")["created_at"]
    listing_seller = listings.set_index("listing_id")["seller_id"]
    view_user = views.set_index("view_id")["user_id"]
    view_listing = views.set_index("view_id")["listing_id"]
    view_time = views.set_index("view_id")["viewed_at"]

    listing_signup_violation = listings.apply(lambda row: row.created_at < user_signup.loc[row.seller_id], axis=1)
    _issue(fatal_errors, "listing_before_signup", "판매글이 판매자 가입 전에 생성됨", _count_mask(listing_signup_violation))
    unverified_seller = listings["seller_id"].map(user_verified).ne(1)
    _issue(
        fatal_errors,
        "listing_seller_not_verified",
        "동네 미인증 사용자가 판매글을 작성함",
        _count_mask(unverified_seller),
        listings.loc[unverified_seller, "listing_id"].tolist(),
    )
    view_signup_violation = views.apply(lambda row: row.viewed_at < user_signup.loc[row.user_id], axis=1)
    _issue(fatal_errors, "view_before_signup", "조회가 사용자 가입 전에 발생", _count_mask(view_signup_violation))
    view_listing_violation = views.apply(lambda row: row.viewed_at < listing_created.loc[row.listing_id], axis=1)
    _issue(fatal_errors, "view_before_listing", "조회가 판매글 생성 전에 발생", _count_mask(view_listing_violation))

    removed_pair_mismatch = favorites["removed_view_id"].isna() != favorites["removed_at"].isna()
    _issue(fatal_errors, "favorite_removed_pair", "removed_view_id와 removed_at 동시 존재 규칙 위반", _count_mask(removed_pair_mismatch))
    favorite_time_violation = favorites["removed_at"].notna() & (favorites["removed_at"] < favorites["favorited_at"])
    _issue(fatal_errors, "favorite_time_order", "찜 취소가 찜 등록보다 빠름", _count_mask(favorite_time_violation))
    favorite_signup_violation = favorites.apply(lambda row: row.favorited_at < user_signup.loc[row.user_id], axis=1)
    _issue(fatal_errors, "favorite_before_signup", "찜이 사용자 가입 전에 발생", _count_mask(favorite_signup_violation))
    favorite_added_mismatch = favorites.apply(
        lambda row: int(view_user.loc[int(row.added_view_id)]) != int(row.user_id)
        or int(view_listing.loc[int(row.added_view_id)]) != int(row.listing_id), axis=1,
    )
    _issue(fatal_errors, "favorite_added_view_mismatch", "added_view_id의 사용자·판매글 불일치", _count_mask(favorite_added_mismatch))
    favorite_before_added_view = favorites.apply(
        lambda row: row.favorited_at < view_time.loc[int(row.added_view_id)], axis=1,
    )
    _issue(fatal_errors, "favorite_before_added_view", "찜이 added_view 조회보다 빠름", _count_mask(favorite_before_added_view))
    removed_rows = favorites[favorites["removed_view_id"].notna()]
    removed_mismatch_count = 0
    removed_before_view_count = 0
    for row in removed_rows.itertuples():
        removed_id = int(row.removed_view_id)
        removed_mismatch_count += int(view_user.loc[removed_id] != row.user_id or view_listing.loc[removed_id] != row.listing_id)
        removed_before_view_count += int(row.removed_at < view_time.loc[removed_id])
    _issue(fatal_errors, "favorite_removed_view_mismatch", "removed_view_id의 사용자·판매글 불일치", removed_mismatch_count)
    _issue(fatal_errors, "favorite_removed_before_view", "찜 취소가 removed_view 조회보다 빠름", removed_before_view_count)
    favorite_seller_violation = favorites.apply(lambda row: int(row.user_id) == int(listing_seller.loc[row.listing_id]), axis=1)
    _issue(fatal_errors, "seller_favorited_own_listing", "판매자가 자신의 판매글을 찜함", _count_mask(favorite_seller_violation))

    duplicate_chat_pairs = chats.duplicated(["listing_id", "buyer_id"], keep=False)
    _issue(fatal_errors, "chat_unique_pair", "chatrooms listing_id+buyer_id 중복", _count_mask(duplicate_chat_pairs))
    chat_seller_mismatch = chats.apply(lambda row: int(row.seller_id) != int(listing_seller.loc[row.listing_id]), axis=1)
    _issue(fatal_errors, "chat_listing_seller_mismatch", "chatrooms.seller_id와 listings.seller_id 불일치", _count_mask(chat_seller_mismatch))
    first_message_order = chats["first_message_at"].notna() & (chats["first_message_at"] < chats["created_at"])
    _issue(fatal_errors, "first_message_order", "첫 메시지가 채팅방 생성보다 빠름", _count_mask(first_message_order))
    response_order = chats["seller_response_at"].notna() & (
        chats["first_message_at"].isna() | (chats["seller_response_at"] < chats["first_message_at"])
    )
    _issue(fatal_errors, "seller_response_order", "판매자 응답이 첫 메시지보다 빠르거나 첫 메시지가 없음", _count_mask(response_order))
    message_state = ~(
        ((chats["message_count"] == 0) & chats["first_message_at"].isna())
        | ((chats["message_count"] >= 1) & chats["first_message_at"].notna())
    )
    _issue(fatal_errors, "chat_message_state", "message_count와 first_message_at 상태 불일치", _count_mask(message_state))
    response_message_count = chats["seller_response_at"].notna() & (chats["message_count"] < 2)
    _issue(fatal_errors, "chat_response_message_count", "판매자 응답이 있는데 message_count가 2 미만", _count_mask(response_message_count))
    minimum_view = views.groupby(["user_id", "listing_id"])["viewed_at"].min().to_dict()
    chat_without_prior_view = chats.apply(
        lambda row: (int(row.buyer_id), int(row.listing_id)) not in minimum_view
        or minimum_view[(int(row.buyer_id), int(row.listing_id))] > row.created_at, axis=1,
    )
    _issue(fatal_errors, "chat_without_prior_view", "채팅 전에 같은 구매자·판매글 조회가 없음", _count_mask(chat_without_prior_view))

    duplicate_transaction_chat = transactions["chatroom_id"].duplicated(keep=False)
    _issue(fatal_errors, "transaction_chat_unique", "한 채팅방에 거래가 둘 이상 존재", _count_mask(duplicate_transaction_chat))
    chat_lookup = chats.set_index("chatroom_id")
    transaction_mismatch = transactions.apply(
        lambda row: any([
            int(row.listing_id) != int(chat_lookup.loc[row.chatroom_id, "listing_id"]),
            int(row.buyer_id) != int(chat_lookup.loc[row.chatroom_id, "buyer_id"]),
            int(row.seller_id) != int(chat_lookup.loc[row.chatroom_id, "seller_id"]),
        ]), axis=1,
    )
    _issue(fatal_errors, "transaction_chat_mismatch", "거래와 연결 채팅방의 판매글·구매자·판매자 불일치", _count_mask(transaction_mismatch))
    transaction_before_message = transactions.apply(
        lambda row: pd.isna(chat_lookup.loc[row.chatroom_id, "first_message_at"])
        or row.started_at < chat_lookup.loc[row.chatroom_id, "first_message_at"], axis=1,
    )
    _issue(fatal_errors, "transaction_before_message", "거래가 첫 메시지 전에 시작됨", _count_mask(transaction_before_message))
    transaction_state = ~(
        ((transactions["transaction_status"] == "거래중") & transactions["completed_at"].isna() & transactions["ended_at"].isna())
        | ((transactions["transaction_status"] == "거래완료") & transactions["completed_at"].notna() & transactions["ended_at"].isna())
        | ((transactions["transaction_status"] == "거래종료") & transactions["completed_at"].isna() & transactions["ended_at"].notna())
    )
    _issue(fatal_errors, "transaction_state", "거래 상태와 완료·종료 시각 불일치", _count_mask(transaction_state))
    completion_order = transactions["completed_at"].notna() & (transactions["completed_at"] < transactions["started_at"])
    ended_order = transactions["ended_at"].notna() & (transactions["ended_at"] < transactions["started_at"])
    _issue(fatal_errors, "transaction_completion_order", "거래 완료가 시작보다 빠름", _count_mask(completion_order))
    _issue(fatal_errors, "transaction_end_order", "거래 종료가 시작보다 빠름", _count_mask(ended_order))

    duplicate_price_time = prices.duplicated(["listing_id", "recorded_at"], keep=False)
    _issue(fatal_errors, "price_history_unique_time", "동일 판매글·기록 시각 중복", _count_mask(duplicate_price_time))
    initial_counts = prices[prices["record_type"] == "initial"].groupby("listing_id").size()
    wrong_initial_count = [listing_id for listing_id in listings["listing_id"] if int(initial_counts.get(listing_id, 0)) != 1]
    _issue(fatal_errors, "price_initial_count", "판매글별 initial 가격 기록이 정확히 하나가 아님", len(wrong_initial_count), wrong_initial_count)
    initial_rows = prices[prices["record_type"] == "initial"]
    initial_time_mismatch = initial_rows.apply(lambda row: row.recorded_at != listing_created.loc[row.listing_id], axis=1)
    _issue(fatal_errors, "price_initial_time", "initial 가격 시각과 판매글 생성 시각 불일치", _count_mask(initial_time_mismatch))
    initial_time_map = initial_rows.set_index("listing_id")["recorded_at"].to_dict()
    changed_rows = prices[prices["record_type"] == "changed"]
    changed_before_initial = changed_rows.apply(lambda row: row.recorded_at <= initial_time_map[row.listing_id], axis=1)
    _issue(fatal_errors, "price_change_order", "changed 가격 기록이 initial보다 빠르거나 같음", _count_mask(changed_before_initial))
    latest_prices = prices.sort_values(["listing_id", "recorded_at", "price_history_id"]).groupby("listing_id").tail(1).set_index("listing_id")["price"]
    current_price_mismatch = listings.apply(lambda row: int(row.price) != int(latest_prices.loc[row.listing_id]), axis=1)
    _issue(fatal_errors, "listing_latest_price", "listings.price와 최신 price_history.price 불일치", _count_mask(current_price_mismatch))
    completed = transactions[transactions["transaction_status"] == "거래완료"]
    completion_map = completed.groupby("listing_id")["completed_at"].min().to_dict()
    changes_after_completion = changed_rows.apply(
        lambda row: row.listing_id in completion_map and row.recorded_at > completion_map[row.listing_id], axis=1,
    )
    _issue(fatal_errors, "price_change_after_completion", "거래 완료 이후 가격 변경 발생", _count_mask(changes_after_completion))
    completed_listing_status = listings[listings["listing_id"].isin(completion_map)]["status"] != "판매완료"
    _issue(fatal_errors, "completed_listing_status", "완료 거래가 있는 판매글의 status가 판매완료가 아님", _count_mask(completed_listing_status))

    latest_activity = _build_latest_activity(tables)
    last_active_mismatch = users.apply(lambda row: row.last_active_at < latest_activity[int(row.user_id)], axis=1)
    _issue(fatal_errors, "last_active_contradiction", "last_active_at이 사용자의 최근 행동보다 빠름", _count_mask(last_active_mismatch))

    expected_counts = {
        "users": (config["user_count"], config["user_count"]),
        "listings": (config["listing_count"], config["listing_count"]),
        "product_views": tuple(config["view_count_range"]),
        "favorites": (4000, 8000),
        "chatrooms": (1000, 3000),
        "transactions": (300, 1000),
    }
    for table, (minimum, maximum) in expected_counts.items():
        if not minimum <= row_counts[table] <= maximum:
            _issue(
                warnings, f"{table}_target_range",
                f"{table} 행 수 {row_counts[table]}가 목표 범위 {minimum}~{maximum} 밖임",
                1, [row_counts[table]],
            )

    anomaly_payload = json.loads((project_root / "generated/anomaly_ground_truth.json").read_text(encoding="utf-8"))
    anomaly_records = anomaly_payload["records"]
    unique_anomaly_rows = {(item["table"], int(item["row_id"])) for item in anomaly_records}
    anomaly_ratio = len(unique_anomaly_rows) / max(1, sum(row_counts.values()))
    requested_anomaly_minimum = 0.03
    requested_anomaly_maximum = 0.06
    if not requested_anomaly_minimum <= anomaly_ratio <= requested_anomaly_maximum:
        _issue(
            warnings, "anomaly_ratio_range",
            f"고유 이상치 행 비율 {anomaly_ratio:.4f}가 요청 범위 0.030~0.060 밖임",
            1, [anomaly_ratio],
        )

    metadata = simulation_metadata or {}
    season_ratio = float(metadata.get("season_related_listing_ratio", 0.0))
    if not 0.20 <= season_ratio <= 0.30:
        _issue(warnings, "season_ratio_range", "계절 관련 판매글 비율이 20~30% 밖임", 1, [season_ratio])

    unique_view_pairs = int(views[["user_id", "listing_id"]].drop_duplicates().shape[0])
    unique_favorite_pairs = int(favorites[["user_id", "listing_id"]].drop_duplicates().shape[0])
    funnel = {
        "view_events": row_counts["product_views"],
        "unique_viewer_listing_pairs": unique_view_pairs,
        "favorite_events": row_counts["favorites"],
        "unique_favorite_pairs": unique_favorite_pairs,
        "chatrooms_created": row_counts["chatrooms"],
        "first_messages": int(chats["first_message_at"].notna().sum()),
        "seller_responses": int(chats["seller_response_at"].notna().sum()),
        "transactions_started": row_counts["transactions"],
        "transactions_completed": int((transactions["transaction_status"] == "거래완료").sum()),
        "view_pair_to_favorite_pair_rate": float(unique_favorite_pairs / max(1, unique_view_pairs)),
        "view_pair_to_chat_rate": float(row_counts["chatrooms"] / max(1, unique_view_pairs)),
        "chat_to_transaction_rate": float(row_counts["transactions"] / max(1, row_counts["chatrooms"])),
        "transaction_completion_rate": float((transactions["transaction_status"] == "거래완료").sum() / max(1, row_counts["transactions"])),
    }

    report = {
        "report_version": config["config_version"],
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed_integrity": not fatal_errors,
        "fatal_error_count": int(sum(item["count"] for item in fatal_errors)),
        "fatal_errors": fatal_errors,
        "warning_count": len(warnings),
        "warnings": warnings,
        "row_counts": row_counts,
        "category_distribution": _distribution(listings["category"]),
        "region_distribution": {
            "users": _distribution(users["region_id"]),
            "listings": _distribution(listings["region_id"]),
        },
        "daily_activity_distribution": _daily_activity(tables),
        "seasonality": {
            "target_season": config["target_season"],
            "configured_strength": float(metadata.get("seasonality_strength", 0.0)),
            "season_related_listing_ratio": season_ratio,
            "listing_context_counts": metadata.get("listing_context_counts", {}),
        },
        "outside_interest_view_ratio": _outside_interest_ratio(project_root, tables),
        "funnel": funnel,
        "price_distribution_by_category": _price_distribution(listings),
        "anomalies": {
            "record_count": len(anomaly_records),
            "unique_row_count": len(unique_anomaly_rows),
            "unique_row_ratio_of_all_rows": anomaly_ratio,
            "configured_rate": float(anomaly_payload["configured_anomaly_rate"]),
            "by_table": dict(Counter(item["table"] for item in anomaly_records)),
            "by_type": dict(Counter(item["anomaly_type"] for item in anomaly_records)),
            "by_origin": dict(Counter(item["injected_or_natural"] for item in anomaly_records)),
        },
        "assumptions": [
            "관찰 종료 시각은 2026-02-28 23:59:59로 해석했다.",
            "users.is_verified는 판매글 작성에 필요한 동네 인증 여부로 해석했다.",
            "계절 관련 비율은 판매글 생성에 사용된 숨은 시나리오 플래그를 기준으로 계산했다.",
            "관심 밖 조회는 주요·보조 관심사에 포함되지 않은 카테고리 조회로 정의했다.",
            "이상치 비율은 7개 테이블 전체 행 중 ground truth에 등장하는 고유 행 비율이다.",
            "분포 목표 이탈은 경고이며 PK·FK·시간 순서 오류만 적재를 차단한다.",
        ],
    }
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return report
