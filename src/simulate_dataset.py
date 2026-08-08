"""설정과 의미 자산을 사용해 7개 MySQL 테이블용 합성 데이터를 생성한다.

LLM 자산은 맥락만 제공한다. 모든 ID, 날짜, 확률적 행동, PK·FK 연결, 시간 순서,
가격 혼합분포와 이상치 주입은 이 모듈의 재현 가능한 Python 시뮬레이션이 담당한다.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TABLE_COLUMNS = {
    "users": ["user_id", "signup_at", "region_id", "manner_score", "is_verified", "last_active_at"],
    "listings": [
        "listing_id", "seller_id", "category", "title", "description", "photo_count",
        "price", "is_negotiable", "region_id", "created_at", "status",
    ],
    "product_views": ["view_id", "user_id", "listing_id", "viewed_at", "source"],
    "favorites": [
        "favorite_id", "user_id", "listing_id", "added_view_id", "favorited_at",
        "removed_view_id", "removed_at",
    ],
    "chatrooms": [
        "chatroom_id", "listing_id", "buyer_id", "seller_id", "created_at",
        "first_message_at", "seller_response_at", "message_count", "chat_status",
    ],
    "transactions": [
        "transaction_id", "listing_id", "chatroom_id", "buyer_id", "seller_id",
        "transaction_status", "started_at", "completed_at", "ended_at",
    ],
    "price_history": ["price_history_id", "listing_id", "price", "record_type", "recorded_at"],
}

PRODUCT_CATALOG = {
    "디지털기기": ["아이패드", "갤럭시 태블릿", "노트북", "기계식 키보드", "무선 이어폰", "모니터", "스마트워치"],
    "생활가전": ["공기청정기", "전기포트", "전자레인지", "로봇청소기", "가습기", "선풍기", "커피머신"],
    "가구 및 인테리어": ["원목 책상", "책장", "식탁 의자", "협탁", "스탠드 조명", "러그", "수납장"],
    "패션": ["겨울 코트", "여름 원피스", "운동화", "가죽 가방", "니트", "청바지", "등산 재킷"],
    "뷰티": ["향수", "헤어 드라이어", "미용기기", "스킨케어 세트", "메이크업 팔레트", "네일 도구"],
    "스포츠 및 레저": ["캠핑 의자", "스키 고글", "자전거", "덤벨 세트", "테니스 라켓", "낚싯대", "요가 매트"],
    "취미 및 게임": ["닌텐도 게임", "보드게임", "피규어", "카메라 렌즈", "뜨개질 세트", "레고", "턴테이블"],
    "도서": ["자격증 교재", "소설 세트", "전공 서적", "그림책 묶음", "에세이", "외국어 교재"],
    "유아 및 아동용품": ["유아 자전거", "아기 의자", "블록 장난감", "유모차", "아동 겨울옷", "놀이 매트"],
    "반려동물용품": ["고양이 이동장", "강아지 계단", "자동 급식기", "반려동물 방석", "캣타워", "산책 가방"],
    "공구": ["전동 드릴", "공구 세트", "사다리", "멀티미터", "작업대", "정원 가위"],
    "생활용품": ["수납 바구니", "보온병", "식기 세트", "빨래 건조대", "전기 담요", "원예 화분"],
    "기타 일반 상품": ["행사 소품", "빈티지 장식품", "여행용 가방", "우산", "기념품", "수집품"],
}


@dataclass
class SimulationResult:
    tables: dict[str, pd.DataFrame]
    persona_profiles: dict[str, Any]
    anomalies: dict[str, Any]
    metadata: dict[str, Any]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _random_datetime(rng: np.random.Generator, start: datetime, end: datetime) -> datetime:
    if end <= start:
        return start
    seconds = int((end - start).total_seconds())
    return start + timedelta(seconds=int(rng.integers(0, seconds + 1)))


def _round_price(value: float, minimum: int, maximum: int) -> int:
    clipped = int(_clamp(value, minimum, maximum))
    unit = 100 if clipped < 100000 else 1000
    return max(minimum, min(maximum, int(round(clipped / unit) * unit)))


def _weighted_choice(rng: np.random.Generator, values: list[Any], weights: list[float]) -> Any:
    probabilities = np.asarray(weights, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return values[int(rng.choice(len(values), p=probabilities))]


def _anomaly_record(
    table: str,
    row_id: int,
    anomaly_type: str,
    reason: str,
    injected_or_natural: str,
    related_user_id: int | None,
    related_listing_id: int | None,
) -> dict[str, Any]:
    return {
        "table": table,
        "row_id": int(row_id),
        "anomaly_type": anomaly_type,
        "reason": reason,
        "injected_or_natural": injected_or_natural,
        "related_user_id": int(related_user_id) if related_user_id is not None else None,
        "related_listing_id": int(related_listing_id) if related_listing_id is not None else None,
    }


def _build_persona_profiles(
    config: dict[str, Any],
    persona_asset: dict[str, Any],
    rng: np.random.Generator,
    user_regions: list[str],
) -> dict[str, Any]:
    templates = persona_asset["templates"]
    profiles: list[dict[str, Any]] = []
    for user_id in range(1, int(config["user_count"]) + 1):
        template = copy.deepcopy(templates[int(rng.integers(0, len(templates)))])
        need_is_active = bool(rng.random() < 0.62)
        profiles.append(
            {
                "profile_id": f"persona_profile_{user_id:04d}",
                "user_id": user_id,
                "template_id": template["template_id"],
                "age_band": template["age_band"],
                "life_stage": template["life_stage"],
                "housing_type": template["housing_type"],
                "occupation_context": template["occupation_context"],
                "primary_interests": template["primary_interests"],
                "secondary_interests": template["secondary_interests"],
                "current_short_need": template["current_short_need"] if need_is_active else "현재 특별한 필요 없음",
                "short_need_categories": template["short_need_categories"] if need_is_active else [],
                "budget_style": template["budget_style"],
                "decision_style": template["decision_style"],
                "negotiation_style": template["negotiation_style"],
                "active_hours": template["active_hours"],
                "travel_range": template["travel_range"],
                "selling_style": template["selling_style"],
                "outside_interest_trigger": template["outside_interest_trigger"],
                "exception_note": template["exception_note"],
                "assigned_region_id": user_regions[user_id - 1],
            }
        )
    return {
        "asset_type": "persona_profiles",
        "version": persona_asset.get("version", "unknown"),
        "persona_assignment_seed": config["seeds"]["persona_assignment_seed"],
        "profiles": profiles,
    }


def _make_users(
    config: dict[str, Any],
    rng: np.random.Generator,
    start: datetime,
) -> tuple[list[dict[str, Any]], list[str]]:
    regions = config["regions"]
    region_ids = [item["region_id"] for item in regions]
    region_weights = [item["weight"] for item in regions]
    user_regions = [
        _weighted_choice(rng, region_ids, region_weights)
        for _ in range(int(config["user_count"]))
    ]
    users: list[dict[str, Any]] = []
    for user_id, region_id in enumerate(user_regions, start=1):
        signup_at = _random_datetime(rng, start - timedelta(days=720), start - timedelta(hours=1))
        manner_score = round(float(np.clip(rng.normal(38.0, 7.5), 5.0, 99.0)), 1)
        users.append(
            {
                "user_id": user_id,
                "signup_at": signup_at,
                "region_id": region_id,
                "manner_score": manner_score,
                "is_verified": int(rng.random() < 0.64),
                "last_active_at": signup_at,
            }
        )
    return users, user_regions


def _choose_category(
    rng: np.random.Generator,
    config: dict[str, Any],
    scenario: dict[str, Any],
    season_related: bool,
) -> str:
    categories = [item["name"] for item in config["categories"]]
    weights = [
        item["weight"] * (item["winter_multiplier"] if season_related else 1.0)
        for item in config["categories"]
    ]
    hint_probability = 0.35 if scenario["context_type"] in {"seasonal", "cross_category"} else 0.20
    if rng.random() < hint_probability and scenario["category_hint"] in categories:
        return scenario["category_hint"]
    return _weighted_choice(rng, categories, weights)


def _render_listing_text(
    rng: np.random.Generator,
    scenario: dict[str, Any],
    category: str,
    region_id: str,
) -> tuple[str, str, int]:
    product = str(rng.choice(PRODUCT_CATALOG[category]))
    condition = scenario["item_condition"]
    reason = scenario["selling_reason"]
    price_note = str(rng.choice(["가격 제안 가능", "급하게 정리", "상태 확인해주세요", "구성품 포함", "천천히 판매"]))
    usage_note = str(rng.choice([
        "실사용 기간은 길지 않습니다.", "보관 기간이 있어 사용 흔적을 확인해 주세요.",
        "기능을 다시 확인했고 정상 작동합니다.", "구매 시기는 오래됐지만 자주 쓰지 않았습니다.",
        "사용 환경이 바뀌어 내놓습니다.",
    ]))
    trade_note = str(rng.choice([
        "직거래를 우선하지만 일정이 맞으면 택배도 가능합니다.", "구매 전 궁금한 점은 채팅 주세요.",
        "사진과 설명을 확인한 뒤 연락 부탁드립니다.", "시간 조율은 비교적 유연합니다.",
        "과도한 가격 제안에는 답하기 어렵습니다.",
    ]))
    values = {
        "product": product,
        "condition": condition,
        "reason": reason,
        "price_note": price_note,
        "region": region_id,
        "usage_note": usage_note,
        "trade_note": trade_note,
    }
    title = scenario["title_pattern"].format(**values)[:100]
    description = scenario["description_pattern"].format(**values)
    style = scenario["description_style"]
    if style in {"상세형", "서사형"}:
        description += " " + scenario["seller_situation"] + ". " + scenario.get("cross_category_context", "")
        description += " 사용하면서 느낀 장점과 불편한 점을 숨기지 않고 적었습니다. 거래 전에 상태를 다시 확인해 드릴게요."
    elif style == "간결형":
        description = f"{product} 판매. {condition}. {trade_note}"
    photo_mean = {"간결형": 2.2, "상세형": 7.0, "대화형": 4.2, "사실 중심형": 5.0, "서사형": 6.0, "목록형": 4.5}.get(style, 4.0)
    photo_count = int(np.clip(rng.poisson(photo_mean), 0, 15))
    return title, description, photo_count


def _make_listings_and_prices(
    config: dict[str, Any],
    listing_asset: dict[str, Any],
    users: list[dict[str, Any]],
    rng: np.random.Generator,
    anomaly_rng: np.random.Generator,
    start: datetime,
    end: datetime,
    anomaly_rate: float,
    seasonality_strength: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    scenarios = listing_asset["scenarios"]
    scenario_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        scenario_groups[scenario["context_type"]].append(scenario)
    price_configs = {item["name"]: item for item in config["categories"]}
    region_map = {item["region_id"]: item for item in config["regions"]}
    listing_count = int(config["listing_count"])
    anomaly_count = max(1, int(round(listing_count * anomaly_rate)))
    anomaly_ids = set(int(value) for value in anomaly_rng.choice(np.arange(1, listing_count + 1), size=anomaly_count, replace=False))
    frequent_change_count = max(1, int(round(listing_count * anomaly_rate * 0.45)))
    frequent_change_ids = set(int(value) for value in anomaly_rng.choice(np.arange(1, listing_count + 1), size=frequent_change_count, replace=False))
    behavior_ranges = config["behavior_probabilities"]
    price_change_probability = float(rng.uniform(*behavior_ranges["price_change_range"]))

    listings: list[dict[str, Any]] = []
    price_history: list[dict[str, Any]] = []
    listing_meta: dict[int, dict[str, Any]] = {}
    anomalies: list[dict[str, Any]] = []
    price_history_id = 1

    for listing_id in range(1, listing_count + 1):
        seller = users[int(rng.integers(0, len(users)))]
        # 판매글을 작성하려는 사용자는 먼저 동네 인증을 완료한다.
        # users.is_verified는 인증 이력 시각이 아닌 관찰 종료 시점의 상태다.
        seller["is_verified"] = 1
        created_at = _random_datetime(rng, start, end - timedelta(days=4))
        season_related = bool(rng.random() < seasonality_strength)
        if season_related:
            group = scenario_groups.get("seasonal") or scenarios
        else:
            context_type = _weighted_choice(rng, ["general", "cross_category", "rare"], [0.70, 0.23, 0.07])
            group = scenario_groups.get(context_type) or scenarios
        scenario = copy.deepcopy(group[int(rng.integers(0, len(group)))])
        category = _choose_category(rng, config, scenario, season_related)
        if rng.random() < 0.76:
            region_id = seller["region_id"]
        elif rng.random() < 0.65 and region_map[seller["region_id"]]["neighbors"]:
            region_id = str(rng.choice(region_map[seller["region_id"]]["neighbors"]))
        else:
            region_id = str(rng.choice(list(region_map)))

        title, description, photo_count = _render_listing_text(rng, scenario, category, region_id)
        price_cfg = price_configs[category]
        initial_price = _round_price(
            float(rng.lognormal(math.log(price_cfg["median_price"]), price_cfg["log_sigma"])),
            int(price_cfg["min_price"]),
            int(price_cfg["max_price"]),
        )
        position = scenario["price_position"]
        position_multiplier = {"bargain": 0.68, "typical": 1.0, "premium": 1.55, "rare": 2.4}.get(position, 1.0)
        initial_price = _round_price(initial_price * position_multiplier, price_cfg["min_price"], price_cfg["max_price"])

        listing_anomaly_type: str | None = None
        if listing_id in anomaly_ids:
            anomaly_variant = listing_id % 4
            if anomaly_variant == 0:
                initial_price = _round_price(initial_price * float(anomaly_rng.uniform(4.0, 9.0)), price_cfg["min_price"], price_cfg["max_price"])
                listing_anomaly_type = "category_relative_price_extreme_high"
            elif anomaly_variant == 1:
                initial_price = _round_price(initial_price * float(anomaly_rng.uniform(0.08, 0.25)), price_cfg["min_price"], price_cfg["max_price"])
                listing_anomaly_type = "category_relative_price_extreme_low"
            elif anomaly_variant == 2:
                photo_count = int(anomaly_rng.integers(0, 2))
                description = f"{title}. 상태는 사진 참고 바랍니다."
                listing_anomaly_type = "minimal_listing_information"
            else:
                initial_price = _round_price(initial_price * float(anomaly_rng.uniform(2.5, 5.0)), price_cfg["min_price"], price_cfg["max_price"])
                description += " 일반 매물보다 드문 구성과 보관 상태를 고려해 가격을 정했습니다."
                listing_anomaly_type = "rare_premium_item"
            anomalies.append(
                _anomaly_record(
                    "listings", listing_id, listing_anomaly_type,
                    "카테고리 기준으로 드물지만 가능한 가격 또는 정보량 극단 사례",
                    "injected", seller["user_id"], listing_id,
                )
            )

        is_negotiable = int(rng.random() < (0.58 if scenario["urgency"] in {"이번 주 안에 정리", "당일 거래 선호"} else 0.40))
        listing = {
            "listing_id": listing_id,
            "seller_id": seller["user_id"],
            "category": category,
            "title": title,
            "description": description,
            "photo_count": photo_count,
            "price": initial_price,
            "is_negotiable": is_negotiable,
            "region_id": region_id,
            "created_at": created_at,
            "status": "판매중",
        }
        price_history.append(
            {
                "price_history_id": price_history_id,
                "listing_id": listing_id,
                "price": initial_price,
                "record_type": "initial",
                "recorded_at": created_at,
            }
        )
        price_history_id += 1

        change_count = 0
        if listing_id in frequent_change_ids:
            change_count = int(anomaly_rng.integers(3, 6))
        elif rng.random() < price_change_probability:
            change_count = 1 + int(rng.random() < 0.16)
        latest_price = initial_price
        changed_records: list[dict[str, Any]] = []
        if change_count > 0:
            available_seconds = int((end - created_at - timedelta(hours=2)).total_seconds())
            if available_seconds > change_count:
                offsets_set: set[int] = set()
                while len(offsets_set) < change_count:
                    offsets_set.add(int(rng.integers(1, available_seconds)))
                offsets = sorted(offsets_set)
                for change_index, offset in enumerate(offsets):
                    change_time = created_at + timedelta(seconds=offset)
                    increase_probability = 0.38 if listing_id in frequent_change_ids else 0.16
                    if rng.random() < increase_probability:
                        multiplier = float(rng.uniform(1.03, 1.22))
                    else:
                        multiplier = float(rng.uniform(0.78, 0.97))
                    latest_price = _round_price(latest_price * multiplier, price_cfg["min_price"], price_cfg["max_price"])
                    record = {
                        "price_history_id": price_history_id,
                        "listing_id": listing_id,
                        "price": latest_price,
                        "record_type": "changed",
                        "recorded_at": change_time,
                    }
                    price_history.append(record)
                    changed_records.append(record)
                    if listing_id in frequent_change_ids:
                        anomalies.append(
                            _anomaly_record(
                                "price_history", price_history_id, "rapid_repeated_price_change",
                                "짧은 관찰 기간에 여러 차례 가격을 조정한 사례",
                                "injected", seller["user_id"], listing_id,
                            )
                        )
                    price_history_id += 1
        listing["price"] = latest_price
        if rng.random() < 0.05:
            listing["status"] = "예약중"
        elif rng.random() < 0.015:
            listing["status"] = "삭제"
        listings.append(listing)
        listing_meta[listing_id] = {
            "scenario_id": scenario["scenario_id"],
            "context_type": scenario["context_type"],
            "season_related": season_related,
            "price_position": position,
            "initial_price": initial_price,
            "latest_price": latest_price,
            "price_decreased": latest_price < initial_price,
            "price_increased": latest_price > initial_price,
            "visibility_score": float(rng.lognormal(0.0, 0.65)) * (1.0 + min(photo_count, 8) * 0.035),
            "latest_price_at": changed_records[-1]["recorded_at"] if changed_records else created_at,
        }

    return listings, price_history, listing_meta, anomalies


def _profile_category(
    rng: np.random.Generator,
    profile: dict[str, Any],
    all_categories: list[str],
) -> tuple[str, bool]:
    draw = rng.random()
    primary = profile["primary_interests"]
    secondary = profile["secondary_interests"]
    short_need = profile["short_need_categories"]
    if draw < 0.53 and primary:
        return str(rng.choice(primary)), False
    if draw < 0.73 and secondary:
        return str(rng.choice(secondary)), False
    if draw < 0.84 and short_need:
        category = str(rng.choice(short_need))
        return category, category not in set(primary + secondary)
    outside = [category for category in all_categories if category not in set(primary + secondary)]
    return str(rng.choice(outside or all_categories)), True


def _make_views(
    config: dict[str, Any],
    users: list[dict[str, Any]],
    profiles: dict[str, Any],
    listings: list[dict[str, Any]],
    listing_meta: dict[int, dict[str, Any]],
    rng: np.random.Generator,
    anomaly_rng: np.random.Generator,
    end: datetime,
    anomaly_rate: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    low, high = config["view_count_range"]
    target_count = int(rng.integers(int(low), int(high) + 1))
    anomaly_view_count = max(1, int(round(target_count * anomaly_rate * 0.90)))
    base_count = target_count - anomaly_view_count
    all_categories = [item["name"] for item in config["categories"]]
    sources = list(config["sources"])
    source_weights = list(config["sources"].values())
    profile_map = {item["user_id"]: item for item in profiles["profiles"]}
    user_map = {item["user_id"]: item for item in users}
    listing_map = {item["listing_id"]: item for item in listings}
    region_neighbors = {item["region_id"]: list(item["neighbors"]) for item in config["regions"]}
    by_category: dict[str, list[int]] = defaultdict(list)
    by_category_region: dict[tuple[str, str], list[int]] = defaultdict(list)
    for listing in listings:
        by_category[listing["category"]].append(listing["listing_id"])
        by_category_region[(listing["category"], listing["region_id"])].append(listing["listing_id"])
    category_probabilities: dict[str, np.ndarray] = {}
    for category, listing_ids in by_category.items():
        weights = np.asarray([listing_meta[listing_id]["visibility_score"] for listing_id in listing_ids], dtype=float)
        category_probabilities[category] = weights / weights.sum()

    activity_weights = rng.lognormal(0.0, 0.70, size=len(users))
    activity_weights = activity_weights / activity_weights.sum()
    views: list[dict[str, Any]] = []
    outside_interest_count = 0

    for _ in range(base_count):
        user_id = int(rng.choice(np.arange(1, len(users) + 1), p=activity_weights))
        user = user_map[user_id]
        profile = profile_map[user_id]
        category, is_outside = _profile_category(rng, profile, all_categories)
        if category not in by_category:
            category = str(rng.choice(all_categories))
        candidate_ids = by_category[category]
        local_draw = rng.random()
        if local_draw < 0.38:
            local_ids = by_category_region.get((category, user["region_id"]), [])
            listing_id = int(rng.choice(local_ids)) if local_ids else int(rng.choice(candidate_ids, p=category_probabilities[category]))
        elif local_draw < 0.58 and region_neighbors[user["region_id"]]:
            neighbor = str(rng.choice(region_neighbors[user["region_id"]]))
            neighbor_ids = by_category_region.get((category, neighbor), [])
            listing_id = int(rng.choice(neighbor_ids)) if neighbor_ids else int(rng.choice(candidate_ids, p=category_probabilities[category]))
        else:
            listing_id = int(rng.choice(candidate_ids, p=category_probabilities[category]))
        listing = listing_map[listing_id]
        viewed_at = _random_datetime(rng, listing["created_at"] + timedelta(seconds=1), end - timedelta(hours=2))
        source = _weighted_choice(rng, sources, source_weights)
        if is_outside and rng.random() < 0.48:
            source = "recommendation"
        views.append(
            {
                "view_id": 0,
                "user_id": user_id,
                "listing_id": listing_id,
                "viewed_at": viewed_at,
                "source": source,
                "_outside_interest": is_outside,
                "_anomaly": False,
            }
        )
        outside_interest_count += int(is_outside)

    remaining = anomaly_view_count
    while remaining > 0:
        burst_size = min(int(anomaly_rng.integers(20, 61)), remaining)
        listing = listings[int(anomaly_rng.integers(0, len(listings)))]
        user = users[int(anomaly_rng.integers(0, len(users)))]
        burst_start = _random_datetime(
            anomaly_rng,
            listing["created_at"] + timedelta(seconds=1),
            end - timedelta(hours=12),
        )
        burst_end = min(end - timedelta(hours=2), burst_start + timedelta(hours=int(anomaly_rng.integers(3, 11))))
        profile = profile_map[user["user_id"]]
        interests = set(profile["primary_interests"] + profile["secondary_interests"])
        is_outside = listing["category"] not in interests
        for _ in range(burst_size):
            views.append(
                {
                    "view_id": 0,
                    "user_id": user["user_id"],
                    "listing_id": listing["listing_id"],
                    "viewed_at": _random_datetime(anomaly_rng, burst_start, burst_end),
                    "source": str(anomaly_rng.choice(["search", "recommendation", "favorite_list", "other"])),
                    "_outside_interest": is_outside,
                    "_anomaly": True,
                }
            )
            outside_interest_count += int(is_outside)
        remaining -= burst_size

    views.sort(key=lambda row: (row["viewed_at"], row["user_id"], row["listing_id"]))
    anomalies: list[dict[str, Any]] = []
    for view_id, view in enumerate(views, start=1):
        view["view_id"] = view_id
        if view.pop("_anomaly"):
            anomalies.append(
                _anomaly_record(
                    "product_views", view_id, "same_day_intensive_repeat_view",
                    "동일 사용자가 같은 판매글을 짧은 시간에 반복 조회한 버스트의 일부",
                    "injected", view["user_id"], view["listing_id"],
                )
            )
        view.pop("_outside_interest")
    return views, anomalies, outside_interest_count / len(views)


def _group_views(views: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for view in views:
        grouped[(view["user_id"], view["listing_id"])].append(view)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["viewed_at"])
    return grouped


def _make_favorites(
    config: dict[str, Any],
    users: list[dict[str, Any]],
    listings: list[dict[str, Any]],
    views_by_pair: dict[tuple[int, int], list[dict[str, Any]]],
    rng: np.random.Generator,
    anomaly_rate: float,
    end: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    user_map = {item["user_id"]: item for item in users}
    listing_map = {item["listing_id"]: item for item in listings}
    probabilities = config["behavior_probabilities"]
    favorite_base = float(rng.uniform(*probabilities["favorite_base_range"]))
    remove_probability = float(rng.uniform(*probabilities["favorite_remove_range"]))
    refavorite_probability = float(rng.uniform(*probabilities["refavorite_after_removal_range"]))
    favorites: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    favorite_id = 1

    for (user_id, listing_id), pair_views in views_by_pair.items():
        listing = listing_map[listing_id]
        if user_id == listing["seller_id"]:
            continue
        user = user_map[user_id]
        repeated_bonus = min(0.10, max(0, len(pair_views) - 1) * 0.018)
        region_bonus = 0.025 if user["region_id"] == listing["region_id"] else 0.0
        quality_bonus = 0.012 if listing["photo_count"] >= 4 and len(listing["description"]) >= 90 else -0.008
        negotiate_bonus = 0.012 if listing["is_negotiable"] else 0.0
        probability = _clamp(
            favorite_base + repeated_bonus + region_bonus + quality_bonus + negotiate_bonus + float(rng.normal(0, 0.025)),
            0.015,
            0.48,
        )
        if rng.random() >= probability:
            continue
        added_index = int(rng.integers(0, max(1, len(pair_views) - 1))) if len(pair_views) > 1 else 0
        added_view = pair_views[added_index]
        favorited_at = added_view["viewed_at"] + timedelta(seconds=int(rng.integers(5, 1201)))
        if favorited_at >= end:
            continue
        removed_view_id: int | None = None
        removed_at: datetime | None = None
        later_views = [view for view in pair_views if view["viewed_at"] > favorited_at]
        chosen_removed_view: dict[str, Any] | None = None
        if later_views and rng.random() < remove_probability:
            chosen_removed_view = later_views[int(rng.integers(0, len(later_views)))]
            removed_view_id = chosen_removed_view["view_id"]
            removed_at = chosen_removed_view["viewed_at"] + timedelta(seconds=int(rng.integers(5, 901)))
            if removed_at >= end:
                removed_view_id = None
                removed_at = None
                chosen_removed_view = None
        favorites.append(
            {
                "favorite_id": favorite_id,
                "user_id": user_id,
                "listing_id": listing_id,
                "added_view_id": added_view["view_id"],
                "favorited_at": favorited_at,
                "removed_view_id": removed_view_id,
                "removed_at": removed_at,
            }
        )
        favorite_id += 1

        if chosen_removed_view is not None:
            views_after_removal = [view for view in pair_views if view["viewed_at"] > removed_at]
            chance = refavorite_probability + (0.25 if len(pair_views) >= 20 else 0.0)
            if views_after_removal and rng.random() < chance:
                readd_view = views_after_removal[int(rng.integers(0, len(views_after_removal)))]
                readd_at = readd_view["viewed_at"] + timedelta(seconds=int(rng.integers(5, 601)))
                if readd_at < end:
                    favorites.append(
                        {
                            "favorite_id": favorite_id,
                            "user_id": user_id,
                            "listing_id": listing_id,
                            "added_view_id": readd_view["view_id"],
                            "favorited_at": readd_at,
                            "removed_view_id": None,
                            "removed_at": None,
                        }
                    )
                    anomalies.append(
                        _anomaly_record(
                            "favorites", favorite_id, "refavorite_after_removal",
                            "찜 취소 뒤 추가 조회를 거쳐 새로운 행으로 다시 찜한 사례",
                            "injected" if rng.random() < 0.75 else "natural", user_id, listing_id,
                        )
                    )
                    favorite_id += 1
    return favorites, anomalies


def _make_chatrooms(
    config: dict[str, Any],
    users: list[dict[str, Any]],
    listings: list[dict[str, Any]],
    listing_meta: dict[int, dict[str, Any]],
    views_by_pair: dict[tuple[int, int], list[dict[str, Any]]],
    favorites: list[dict[str, Any]],
    rng: np.random.Generator,
    anomaly_rng: np.random.Generator,
    anomaly_rate: float,
    end: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    user_map = {item["user_id"]: item for item in users}
    listing_map = {item["listing_id"]: item for item in listings}
    favorite_pairs = {(row["user_id"], row["listing_id"]) for row in favorites}
    probabilities = config["behavior_probabilities"]
    chat_base = float(rng.uniform(*probabilities["chat_base_range"]))
    first_message_probability = float(rng.uniform(*probabilities["first_message_range"]))
    response_probability = float(rng.uniform(*probabilities["seller_response_range"]))
    chatrooms: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    chatroom_id = 1

    for (buyer_id, listing_id), pair_views in views_by_pair.items():
        listing = listing_map[listing_id]
        if buyer_id == listing["seller_id"]:
            continue
        buyer = user_map[buyer_id]
        has_favorite = (buyer_id, listing_id) in favorite_pairs
        repeat_bonus = min(0.075, max(0, len(pair_views) - 1) * 0.012)
        probability = chat_base
        probability += 0.085 if has_favorite else 0.0
        probability += repeat_bonus
        probability += 0.018 if buyer["region_id"] == listing["region_id"] else 0.0
        probability += 0.012 if listing["is_negotiable"] else 0.0
        probability += 0.016 if listing_meta[listing_id]["price_decreased"] else 0.0
        probability += 0.008 if listing["price"] >= listing_meta[listing_id]["initial_price"] * 1.5 else 0.0
        probability += float(rng.normal(0, 0.022))
        if rng.random() >= _clamp(probability, 0.008, 0.38):
            continue
        view = pair_views[int(rng.integers(max(0, len(pair_views) // 2), len(pair_views)))]
        created_at = view["viewed_at"] + timedelta(seconds=int(rng.integers(60, 7201)))
        if created_at >= end - timedelta(minutes=10):
            continue

        first_message_at: datetime | None = None
        seller_response_at: datetime | None = None
        message_count = 0
        chat_status = "진행중"
        chat_anomaly: tuple[str, str] | None = None
        if rng.random() < first_message_probability:
            first_message_at = created_at + timedelta(seconds=int(rng.integers(10, 3601)))
            if first_message_at >= end:
                first_message_at = None
            else:
                message_count = int(1 + rng.poisson(2.2))
                if rng.random() < response_probability:
                    if anomaly_rng.random() < anomaly_rate * 0.55:
                        response_delay = timedelta(days=float(anomaly_rng.uniform(2.0, 6.0)))
                        chat_anomaly = ("multi_day_seller_response", "판매자의 첫 응답까지 며칠이 걸린 사례")
                    else:
                        response_delay = timedelta(minutes=float(np.clip(rng.exponential(210), 1, 1440)))
                    candidate_response = first_message_at + response_delay
                    if candidate_response < end:
                        seller_response_at = candidate_response
                        message_count = max(2, int(2 + rng.poisson(7.5)))
                if anomaly_rng.random() < anomaly_rate * 0.55:
                    message_count = int(anomaly_rng.integers(100, 301))
                    chat_anomaly = ("very_long_negotiation", "100~300개 메시지가 오간 긴 협상 또는 반복 문의")
                chat_status = str(rng.choice(["진행중", "종료", "차단"], p=[0.54, 0.42, 0.04]))
        else:
            chat_status = str(rng.choice(["진행중", "종료"], p=[0.35, 0.65]))

        chatrooms.append(
            {
                "chatroom_id": chatroom_id,
                "listing_id": listing_id,
                "buyer_id": buyer_id,
                "seller_id": listing["seller_id"],
                "created_at": created_at,
                "first_message_at": first_message_at,
                "seller_response_at": seller_response_at,
                "message_count": message_count,
                "chat_status": chat_status,
            }
        )
        if chat_anomaly is not None:
            anomalies.append(
                _anomaly_record(
                    "chatrooms", chatroom_id, chat_anomaly[0], chat_anomaly[1],
                    "injected", buyer_id, listing_id,
                )
            )
        chatroom_id += 1
    return chatrooms, anomalies


def _make_transactions(
    config: dict[str, Any],
    listings: list[dict[str, Any]],
    listing_meta: dict[int, dict[str, Any]],
    chatrooms: list[dict[str, Any]],
    rng: np.random.Generator,
    anomaly_rng: np.random.Generator,
    anomaly_rate: float,
    end: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    listing_map = {item["listing_id"]: item for item in listings}
    probabilities = config["behavior_probabilities"]
    transaction_base = float(rng.uniform(*probabilities["transaction_base_range"]))
    completion_probability = float(rng.uniform(*probabilities["completion_given_transaction_range"]))
    ended_probability = float(rng.uniform(*probabilities["ended_given_transaction_range"]))
    transactions: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    completed_listings: set[int] = set()
    transaction_id = 1

    valid_chats = [chat for chat in chatrooms if chat["first_message_at"] is not None]
    valid_chats.sort(key=lambda row: row["first_message_at"])
    for chat in valid_chats:
        listing_id = chat["listing_id"]
        if listing_id in completed_listings:
            continue
        probability = transaction_base
        probability += 0.11 if chat["seller_response_at"] is not None else -0.05
        probability += min(0.08, chat["message_count"] / 2500)
        probability += float(rng.normal(0, 0.025))
        if rng.random() >= _clamp(probability, 0.05, 0.58):
            continue
        started_at = chat["first_message_at"] + timedelta(minutes=float(rng.uniform(5, 2880)))
        if started_at >= end:
            continue
        completed_at: datetime | None = None
        ended_at: datetime | None = None
        transaction_status = "거래중"
        transaction_anomaly: tuple[str, str] | None = None
        status_draw = rng.random()
        if status_draw < completion_probability:
            latest_price_at = listing_meta[listing_id]["latest_price_at"]
            if anomaly_rng.random() < anomaly_rate * 0.65 and latest_price_at <= started_at:
                candidate = started_at + timedelta(minutes=float(anomaly_rng.uniform(3, 25)))
                transaction_anomaly = ("very_fast_completion", "채팅 후 거래가 매우 빠르게 완료된 사례")
            else:
                candidate = started_at + timedelta(hours=float(rng.uniform(2, 240)))
                candidate = max(candidate, latest_price_at + timedelta(minutes=1))
            if candidate < end:
                completed_at = candidate
                transaction_status = "거래완료"
                completed_listings.add(listing_id)
                listing_map[listing_id]["status"] = "판매완료"
            else:
                transaction_status = "거래중"
        elif status_draw < completion_probability + ended_probability:
            candidate = started_at + timedelta(hours=float(rng.uniform(1, 240)))
            if candidate < end:
                ended_at = candidate
                transaction_status = "거래종료"
        elif (end - started_at).days >= 30 and anomaly_rng.random() < anomaly_rate:
            transaction_anomaly = ("long_running_transaction", "수십 일 동안 거래중 상태가 유지된 사례")

        transactions.append(
            {
                "transaction_id": transaction_id,
                "listing_id": listing_id,
                "chatroom_id": chat["chatroom_id"],
                "buyer_id": chat["buyer_id"],
                "seller_id": chat["seller_id"],
                "transaction_status": transaction_status,
                "started_at": started_at,
                "completed_at": completed_at,
                "ended_at": ended_at,
            }
        )
        if transaction_anomaly is not None:
            anomalies.append(
                _anomaly_record(
                    "transactions", transaction_id, transaction_anomaly[0], transaction_anomaly[1],
                    "injected", chat["buyer_id"], listing_id,
                )
            )
        transaction_id += 1

    active_listing_ids = {
        row["listing_id"] for row in transactions if row["transaction_status"] == "거래중"
    }
    for listing_id in active_listing_ids - completed_listings:
        if listing_map[listing_id]["status"] == "판매중":
            listing_map[listing_id]["status"] = "예약중"
    return transactions, anomalies


def _update_last_active(
    users: list[dict[str, Any]],
    listings: list[dict[str, Any]],
    views: list[dict[str, Any]],
    favorites: list[dict[str, Any]],
    chatrooms: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
) -> None:
    latest = {user["user_id"]: user["signup_at"] for user in users}

    def update(user_id: int, value: datetime | None) -> None:
        if value is not None and value > latest[user_id]:
            latest[user_id] = value

    for listing in listings:
        update(listing["seller_id"], listing["created_at"])
    for view in views:
        update(view["user_id"], view["viewed_at"])
    for favorite in favorites:
        update(favorite["user_id"], favorite["favorited_at"])
        update(favorite["user_id"], favorite["removed_at"])
    for chat in chatrooms:
        update(chat["buyer_id"], chat["created_at"])
        update(chat["buyer_id"], chat["first_message_at"])
        update(chat["seller_id"], chat["seller_response_at"])
    for transaction in transactions:
        final_time = transaction["completed_at"] or transaction["ended_at"] or transaction["started_at"]
        update(transaction["buyer_id"], final_time)
        update(transaction["seller_id"], final_time)
    for user in users:
        user["last_active_at"] = latest[user["user_id"]]


def _add_natural_anomalies(
    tables: dict[str, pd.DataFrame],
    listing_meta: dict[int, dict[str, Any]],
    anomalies: list[dict[str, Any]],
) -> None:
    listings = tables["listings"].set_index("listing_id")
    views = tables["product_views"].groupby("listing_id").size()
    favorites = tables["favorites"].groupby("listing_id").size() if not tables["favorites"].empty else pd.Series(dtype=int)
    chats = tables["chatrooms"].groupby("listing_id").size() if not tables["chatrooms"].empty else pd.Series(dtype=int)
    transactions = tables["transactions"]

    for listing_id in views.sort_values(ascending=False).head(20).index:
        if int(favorites.get(listing_id, 0)) == 0 and int(chats.get(listing_id, 0)) == 0:
            seller_id = int(listings.loc[listing_id, "seller_id"])
            anomalies.append(
                _anomaly_record(
                    "listings", int(listing_id), "high_views_without_favorite_or_chat",
                    "조회는 매우 많지만 찜과 채팅이 자연스럽게 발생하지 않은 판매글",
                    "natural", seller_id, int(listing_id),
                )
            )

    completed_ids = set(
        transactions.loc[transactions["transaction_status"] == "거래완료", "listing_id"].astype(int)
    ) if not transactions.empty else set()
    for listing_id in sorted(completed_ids):
        if listing_meta[listing_id]["price_increased"]:
            seller_id = int(listings.loc[listing_id, "seller_id"])
            anomalies.append(
                _anomaly_record(
                    "listings", listing_id, "completed_after_price_increase",
                    "가격을 인상했는데도 거래가 완료된 반대 사례",
                    "natural", seller_id, listing_id,
                )
            )


def simulate_dataset(
    config: dict[str, Any],
    persona_asset: dict[str, Any],
    listing_asset: dict[str, Any],
) -> SimulationResult:
    """전체 데이터셋을 한 번 생성해 DataFrame과 숨은 자산을 반환한다."""

    start = datetime.fromisoformat(config["observation_start"])
    end = datetime.fromisoformat(config["observation_end"]) + timedelta(hours=23, minutes=59, seconds=59)
    seeds = config["seeds"]
    persona_rng = np.random.default_rng(seeds["persona_assignment_seed"])
    listing_rng = np.random.default_rng(seeds["listing_generation_seed"])
    exposure_rng = np.random.default_rng(seeds["exposure_seed"])
    behavior_rng = np.random.default_rng(seeds["behavior_seed"])
    anomaly_rng = np.random.default_rng(seeds["anomaly_seed"])
    seasonality_strength = float(listing_rng.uniform(*config["seasonality_strength_range"]))
    anomaly_rate = float(anomaly_rng.uniform(*config["anomaly_rate_range"]))

    users, user_regions = _make_users(config, persona_rng, start)
    persona_profiles = _build_persona_profiles(config, persona_asset, persona_rng, user_regions)
    listings, price_history, listing_meta, listing_anomalies = _make_listings_and_prices(
        config, listing_asset, users, listing_rng, anomaly_rng, start, end,
        anomaly_rate, seasonality_strength,
    )
    views, view_anomalies, outside_interest_ratio = _make_views(
        config, users, persona_profiles, listings, listing_meta, exposure_rng,
        anomaly_rng, end, anomaly_rate,
    )
    views_by_pair = _group_views(views)
    favorites, favorite_anomalies = _make_favorites(
        config, users, listings, views_by_pair, behavior_rng, anomaly_rate, end,
    )
    chatrooms, chat_anomalies = _make_chatrooms(
        config, users, listings, listing_meta, views_by_pair, favorites,
        behavior_rng, anomaly_rng, anomaly_rate, end,
    )
    transactions, transaction_anomalies = _make_transactions(
        config, listings, listing_meta, chatrooms, behavior_rng, anomaly_rng,
        anomaly_rate, end,
    )
    _update_last_active(users, listings, views, favorites, chatrooms, transactions)

    table_rows = {
        "users": users,
        "listings": listings,
        "product_views": views,
        "favorites": favorites,
        "chatrooms": chatrooms,
        "transactions": transactions,
        "price_history": price_history,
    }
    tables = {
        name: pd.DataFrame(rows, columns=TABLE_COLUMNS[name])
        for name, rows in table_rows.items()
    }
    anomalies = (
        listing_anomalies + view_anomalies + favorite_anomalies
        + chat_anomalies + transaction_anomalies
    )
    _add_natural_anomalies(tables, listing_meta, anomalies)
    anomaly_payload = {
        "version": config["config_version"],
        "configured_anomaly_rate": anomaly_rate,
        "records": anomalies,
    }
    actual_season_ratio = sum(1 for meta in listing_meta.values() if meta["season_related"]) / len(listing_meta)
    metadata = {
        "seasonality_strength": seasonality_strength,
        "season_related_listing_ratio": actual_season_ratio,
        "configured_anomaly_rate": anomaly_rate,
        "outside_interest_view_ratio": outside_interest_ratio,
        "listing_context_counts": {
            context: sum(1 for meta in listing_meta.values() if meta["context_type"] == context)
            for context in ["general", "cross_category", "seasonal", "rare"]
        },
    }
    return SimulationResult(tables, persona_profiles, anomaly_payload, metadata)


def write_simulation_outputs(result: SimulationResult, project_root: Path) -> None:
    """7개 CSV와 숨은 페르소나·이상치 JSON을 UTF-8로 저장한다."""

    csv_dir = project_root / "generated/csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in result.tables.items():
        output = frame.copy()
        for column in output.columns:
            if column.endswith("_at"):
                output[column] = pd.to_datetime(output[column]).dt.strftime("%Y-%m-%d %H:%M:%S")
        output.to_csv(csv_dir / f"{table_name}.csv", index=False, encoding="utf-8")
    (project_root / "generated/persona_profiles.json").write_text(
        json.dumps(result.persona_profiles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (project_root / "generated/anomaly_ground_truth.json").write_text(
        json.dumps(result.anomalies, ensure_ascii=False, indent=2), encoding="utf-8"
    )
