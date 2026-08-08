"""합성 CSV를 사용자-판매글 단위의 Tableau 대시보드 데이터로 변환한다."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT_DIR / "generated" / "csv"
OUTPUT_DIR = ROOT_DIR / "tableau"
OUTPUT_PATH = OUTPUT_DIR / "tableau_dashboard_data.csv"


def _read_csv(name: str, date_columns: list[str]) -> pd.DataFrame:
    """생성된 CSV를 읽고 지정한 날짜 컬럼을 datetime으로 변환한다."""
    frame = pd.read_csv(CSV_DIR / name)
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def build_tableau_dataset() -> pd.DataFrame:
    """한 행이 사용자 1명과 판매글 1개의 행동 여정인 데이터셋을 만든다."""
    users = _read_csv("users.csv", ["signup_at", "last_active_at"])
    listings = _read_csv("listings.csv", ["created_at"])
    views = _read_csv("product_views.csv", ["viewed_at"])
    favorites = _read_csv("favorites.csv", ["favorited_at", "removed_at"])
    chatrooms = _read_csv(
        "chatrooms.csv",
        ["created_at", "first_message_at", "seller_response_at"],
    )
    transactions = _read_csv(
        "transactions.csv",
        ["started_at", "completed_at", "ended_at"],
    )
    price_history = _read_csv("price_history.csv", ["recorded_at"])

    # 판매자 자신의 조회는 행동 퍼널에서 제외한다.
    valid_views = views.merge(
        listings[["listing_id", "seller_id"]],
        on="listing_id",
        how="left",
        validate="many_to_one",
    )
    valid_views = valid_views[valid_views["user_id"] != valid_views["seller_id"]]

    view_pair = (
        valid_views.groupby(["user_id", "listing_id"], as_index=False)
        .agg(
            user_view_count=("view_id", "size"),
            first_view_at=("viewed_at", "min"),
            last_view_at=("viewed_at", "max"),
        )
    )

    favorite_pair = (
        favorites.groupby(["user_id", "listing_id"], as_index=False)
        .agg(
            favorite_cycle_count=("favorite_id", "size"),
            favorite_remove_count=("removed_at", lambda x: int(x.notna().sum())),
            is_currently_favorited=("removed_at", lambda x: int(x.isna().any())),
            first_favorite_at=("favorited_at", "min"),
        )
    )

    chat_with_transaction = chatrooms.merge(
        transactions,
        on=["chatroom_id", "listing_id", "seller_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_transaction"),
    )
    chat_pair = (
        chat_with_transaction.groupby(
            ["buyer_id", "listing_id"], as_index=False, dropna=False
        )
        .agg(
            chatroom_id=("chatroom_id", "first"),
            chatroom_created_at=("created_at", "min"),
            first_message_at=("first_message_at", "min"),
            seller_response_at=("seller_response_at", "min"),
            message_count=("message_count", "max"),
            chat_status=("chat_status", "first"),
            transaction_id=("transaction_id", "first"),
            transaction_status=("transaction_status", "first"),
            transaction_started_at=("started_at", "min"),
            transaction_completed_at=("completed_at", "min"),
            transaction_ended_at=("ended_at", "min"),
        )
        .rename(columns={"buyer_id": "user_id"})
    )

    listing_features = listings[
        [
            "listing_id",
            "seller_id",
            "category",
            "price",
            "photo_count",
            "description",
            "is_negotiable",
            "region_id",
            "created_at",
            "status",
        ]
    ].copy()
    listing_features["description_length"] = (
        listing_features["description"].fillna("").str.len()
    )
    listing_features = listing_features.drop(columns="description").rename(
        columns={
            "price": "current_price",
            "region_id": "listing_region_id",
            "created_at": "listing_created_at",
            "status": "listing_status",
        }
    )

    user_features = users[["user_id", "region_id"]].rename(
        columns={"region_id": "user_region_id"}
    )
    seller_features = users[["user_id", "manner_score"]].rename(
        columns={
            "user_id": "seller_id",
            "manner_score": "seller_manner_score",
        }
    )

    dashboard = (
        view_pair.merge(user_features, on="user_id", how="left", validate="many_to_one")
        .merge(listing_features, on="listing_id", how="left", validate="many_to_one")
        .merge(seller_features, on="seller_id", how="left", validate="many_to_one")
        .merge(
            favorite_pair,
            on=["user_id", "listing_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            chat_pair,
            on=["user_id", "listing_id"],
            how="left",
            validate="one_to_one",
        )
    )

    count_columns = [
        "favorite_cycle_count",
        "favorite_remove_count",
        "is_currently_favorited",
        "message_count",
    ]
    dashboard[count_columns] = dashboard[count_columns].fillna(0).astype(int)

    # 찜 이후부터 채팅 전까지의 재조회 수를 계산한다.
    favorite_times = dashboard.loc[
        dashboard["first_favorite_at"].notna(),
        ["user_id", "listing_id", "first_favorite_at", "first_message_at"],
    ]
    revisit_events = valid_views.merge(
        favorite_times,
        on=["user_id", "listing_id"],
        how="inner",
        validate="many_to_one",
    )
    revisit_events = revisit_events[
        (revisit_events["viewed_at"] > revisit_events["first_favorite_at"])
        & (
            revisit_events["first_message_at"].isna()
            | (revisit_events["viewed_at"] <= revisit_events["first_message_at"])
        )
    ]
    revisit_count = (
        revisit_events.groupby(["user_id", "listing_id"])
        .size()
        .rename("post_favorite_revisit_count")
        .reset_index()
    )
    dashboard = dashboard.merge(
        revisit_count,
        on=["user_id", "listing_id"],
        how="left",
        validate="one_to_one",
    )
    dashboard["post_favorite_revisit_count"] = (
        dashboard["post_favorite_revisit_count"].fillna(0).astype(int)
    )

    # 찜 이전 반복 조회 수를 계산한다. 찜하지 않은 행은 결측값으로 둔다.
    pre_favorite_views = valid_views.merge(
        favorite_times[["user_id", "listing_id", "first_favorite_at"]],
        on=["user_id", "listing_id"],
        how="inner",
        validate="many_to_one",
    )
    pre_favorite_views = pre_favorite_views[
        pre_favorite_views["viewed_at"] <= pre_favorite_views["first_favorite_at"]
    ]
    pre_favorite_count = (
        pre_favorite_views.groupby(["user_id", "listing_id"])
        .size()
        .sub(1)
        .clip(lower=0)
        .rename("repeat_view_count_before_favorite")
        .reset_index()
    )
    dashboard = dashboard.merge(
        pre_favorite_count,
        on=["user_id", "listing_id"],
        how="left",
        validate="one_to_one",
    )
    dashboard["repeat_view_count_before_favorite"] = dashboard[
        "repeat_view_count_before_favorite"
    ].astype("Int64")

    # 최초 찜 시점까지 가격이 인하됐는지 계산한다.
    initial_price = (
        price_history[price_history["record_type"] == "initial"]
        .sort_values(["listing_id", "recorded_at"])
        .drop_duplicates("listing_id", keep="first")
        [["listing_id", "price"]]
        .rename(columns={"price": "initial_price"})
    )
    price_candidates = favorite_times[
        ["user_id", "listing_id", "first_favorite_at"]
    ].merge(price_history, on="listing_id", how="left", validate="many_to_many")
    price_candidates = price_candidates[
        price_candidates["recorded_at"] <= price_candidates["first_favorite_at"]
    ]
    price_at_favorite = (
        price_candidates.sort_values("recorded_at")
        .drop_duplicates(["user_id", "listing_id"], keep="last")
        [["user_id", "listing_id", "price"]]
        .rename(columns={"price": "price_at_favorite"})
        .merge(initial_price, on="listing_id", how="left", validate="many_to_one")
    )
    price_at_favorite["price_decreased_before_favorite"] = (
        price_at_favorite["price_at_favorite"] < price_at_favorite["initial_price"]
    ).astype(int)
    dashboard = dashboard.merge(
        price_at_favorite,
        on=["user_id", "listing_id"],
        how="left",
        validate="one_to_one",
    )
    dashboard["price_decreased_before_favorite"] = dashboard[
        "price_decreased_before_favorite"
    ].astype("Int64")

    category_median = listings.groupby("category")["price"].median()
    dashboard["category_median_price"] = dashboard["category"].map(category_median)
    dashboard["relative_price_index"] = (
        dashboard["current_price"] / dashboard["category_median_price"]
    ).round(4)
    dashboard["same_region"] = (
        dashboard["user_region_id"] == dashboard["listing_region_id"]
    ).astype(int)

    has_favorite = dashboard["first_favorite_at"].notna()
    has_chat = dashboard["first_message_at"].notna()
    has_revisit = dashboard["post_favorite_revisit_count"] > 0
    favorite_before_chat = (
        has_favorite
        & has_chat
        & (dashboard["first_favorite_at"] <= dashboard["first_message_at"])
    )
    favorite_journey = has_favorite & (
        ~has_chat | (dashboard["first_favorite_at"] <= dashboard["first_message_at"])
    )

    dashboard["journey_group"] = np.select(
        [
            ~has_favorite & ~has_chat & (dashboard["user_view_count"] == 1),
            ~has_favorite & ~has_chat & (dashboard["user_view_count"] >= 2),
            has_favorite & ~has_chat & ~has_revisit,
            has_favorite & ~has_chat & has_revisit,
            favorite_before_chat & ~has_revisit,
            favorite_before_chat & has_revisit,
            has_chat,
        ],
        [
            "view_only",
            "repeat_view_only",
            "favorite_stalled_no_revisit",
            "favorite_stalled_revisit",
            "favorite_to_chat_no_revisit",
            "favorite_revisit_to_chat",
            "direct_chat",
        ],
        default="unclassified",
    )
    journey_labels = {
        "view_only": "조회만",
        "repeat_view_only": "조회 → 재조회",
        "favorite_stalled_no_revisit": "조회 → 찜 → 채팅 없음",
        "favorite_stalled_revisit": "조회 → 찜 → 재조회 → 채팅 없음",
        "favorite_to_chat_no_revisit": "조회 → 찜 → 채팅",
        "favorite_revisit_to_chat": "조회 → 찜 → 재조회 → 채팅",
        "direct_chat": "조회 → 바로 채팅",
    }
    dashboard["journey_label"] = dashboard["journey_group"].map(journey_labels)

    dashboard["journey_id"] = (
        dashboard["user_id"].astype(str) + "_" + dashboard["listing_id"].astype(str)
    )
    dashboard["has_favorite"] = has_favorite.astype(int)
    dashboard["is_favorite_journey"] = favorite_journey.astype(int)
    dashboard["has_revisit_after_favorite"] = has_revisit.astype(int)
    dashboard["has_chatroom"] = dashboard["chatroom_created_at"].notna().astype(int)
    dashboard["has_chat"] = has_chat.astype(int)
    dashboard["favorite_chat_converted"] = favorite_before_chat.astype(int)
    dashboard["is_direct_chat"] = dashboard["journey_group"].eq("direct_chat").astype(int)
    dashboard["has_transaction"] = dashboard["transaction_id"].notna().astype(int)
    dashboard["transaction_completed"] = (
        dashboard["transaction_status"] == "거래완료"
    ).astype(int)
    dashboard["transaction_ended"] = (
        dashboard["transaction_status"] == "거래종료"
    ).astype(int)

    dashboard["chat_path"] = np.select(
        [dashboard["is_direct_chat"].eq(1), favorite_before_chat],
        ["직접 채팅", "찜 후 채팅"],
        default="채팅 없음",
    )
    dashboard["favorite_outcome"] = np.select(
        [
            ~has_favorite,
            favorite_before_chat,
            has_favorite & has_chat & ~favorite_before_chat,
        ],
        ["찜 없음", "찜 후 채팅 전환", "채팅 후 찜 (분석 제외)"],
        default="찜 후 채팅 미전환",
    )
    dashboard["revisit_outcome"] = np.select(
        [~(has_favorite & has_revisit), favorite_before_chat],
        ["해당 없음", "재조회 후 채팅"],
        default="재조회 후 채팅 없음",
    )
    dashboard["revisit_bucket"] = np.select(
        [
            dashboard["post_favorite_revisit_count"].eq(0),
            dashboard["post_favorite_revisit_count"].eq(1),
            dashboard["post_favorite_revisit_count"].eq(2),
        ],
        ["재조회 없음", "1회", "2회"],
        default="3회 이상",
    )
    dashboard["transaction_status"] = dashboard["transaction_status"].fillna(
        "거래 미진입"
    )

    output_columns = [
        "journey_id",
        "user_id",
        "listing_id",
        "seller_id",
        "journey_group",
        "journey_label",
        "chat_path",
        "favorite_outcome",
        "revisit_outcome",
        "revisit_bucket",
        "has_favorite",
        "is_favorite_journey",
        "has_revisit_after_favorite",
        "has_chatroom",
        "has_chat",
        "favorite_chat_converted",
        "is_direct_chat",
        "has_transaction",
        "transaction_completed",
        "transaction_ended",
        "user_view_count",
        "repeat_view_count_before_favorite",
        "post_favorite_revisit_count",
        "favorite_cycle_count",
        "favorite_remove_count",
        "is_currently_favorited",
        "category",
        "current_price",
        "category_median_price",
        "relative_price_index",
        "photo_count",
        "description_length",
        "is_negotiable",
        "same_region",
        "seller_manner_score",
        "user_region_id",
        "listing_region_id",
        "listing_status",
        "initial_price",
        "price_at_favorite",
        "price_decreased_before_favorite",
        "message_count",
        "chat_status",
        "transaction_status",
        "listing_created_at",
        "first_view_at",
        "last_view_at",
        "first_favorite_at",
        "chatroom_created_at",
        "first_message_at",
        "seller_response_at",
        "transaction_started_at",
        "transaction_completed_at",
        "transaction_ended_at",
    ]
    dashboard = dashboard[output_columns].sort_values(
        ["listing_id", "user_id"]
    ).reset_index(drop=True)

    if dashboard.duplicated(["user_id", "listing_id"]).any():
        raise ValueError("사용자-판매글 조합이 중복되었습니다.")
    if dashboard["journey_group"].eq("unclassified").any():
        raise ValueError("분류되지 않은 행동 여정이 존재합니다.")
    if dashboard["journey_label"].isna().any():
        raise ValueError("행동 여정 한글명이 누락되었습니다.")
    favorite_groups = {
        "favorite_stalled_no_revisit",
        "favorite_stalled_revisit",
        "favorite_to_chat_no_revisit",
        "favorite_revisit_to_chat",
    }
    expected_favorite_journey = dashboard["journey_group"].isin(favorite_groups).astype(int)
    if not dashboard["is_favorite_journey"].equals(expected_favorite_journey):
        raise ValueError("찜 분석 대상과 행동 여정 분류가 일치하지 않습니다.")
    if (dashboard["has_transaction"] > dashboard["has_chat"]).any():
        raise ValueError("채팅 없이 거래가 시작된 행이 존재합니다.")

    return dashboard


def main() -> None:
    """Tableau CSV를 생성하고 핵심 검증 수치를 출력한다."""
    dashboard = build_tableau_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dashboard.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"전체 행동 여정: {len(dashboard):,}건")
    print(f"찜 이벤트 보유: {dashboard['has_favorite'].sum():,}건")
    print(f"찜→채팅 분석 대상: {dashboard['is_favorite_journey'].sum():,}건")
    print(f"전체 채팅: {dashboard['has_chat'].sum():,}건")
    print(f"찜 후 채팅: {dashboard['favorite_chat_converted'].sum():,}건")
    print(
        "찜 후 재조회: "
        f"{dashboard['has_revisit_after_favorite'].sum():,}건 "
        f"(채팅 {dashboard['revisit_outcome'].eq('재조회 후 채팅').sum():,}건 / "
        f"미전환 {dashboard['revisit_outcome'].eq('재조회 후 채팅 없음').sum():,}건)"
    )
    print(f"거래 완료: {dashboard['transaction_completed'].sum():,}건")


if __name__ == "__main__":
    main()
