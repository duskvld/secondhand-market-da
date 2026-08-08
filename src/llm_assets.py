"""LLM 의미 자산을 캐시 우선 방식으로 준비한다.

API 키가 없으면 Codex가 작성한 조합 규칙으로 50개 이상의 페르소나 템플릿과
60개 이상의 판매 상황을 즉시 만든다. 명시적으로 갱신을 요청하고 ``.env``에
``OPENAI_API_KEY``와 ``OPENAI_MODEL``이 모두 있을 때만 OpenAI Structured Outputs를
호출한다. 어느 경우에도 LLM은 행동 결과를 결정하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel


class PersonaTemplate(BaseModel):
    """행동 확률이 아닌 의미 범주로 표현한 사용자 템플릿."""

    template_id: str
    age_band: str
    life_stage: str
    housing_type: str
    occupation_context: str
    primary_interests: list[str]
    secondary_interests: list[str]
    current_short_need: str
    short_need_categories: list[str]
    budget_style: str
    decision_style: str
    negotiation_style: str
    active_hours: str
    travel_range: str
    selling_style: str
    outside_interest_trigger: str
    exception_note: str


class PersonaTemplateBatch(BaseModel):
    templates: list[PersonaTemplate]


class ListingScenario(BaseModel):
    """판매 결과를 포함하지 않는 판매글 의미 상황."""

    scenario_id: str
    context_type: str
    category_hint: str
    seller_situation: str
    selling_reason: str
    item_condition: str
    urgency: str
    price_position: str
    title_style: str
    description_style: str
    title_pattern: str
    description_pattern: str
    season_related: bool
    cross_category_context: str
    rare_case: str


class ListingScenarioBatch(BaseModel):
    scenarios: list[ListingScenario]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fallback_personas(config: dict[str, Any]) -> dict[str, Any]:
    count = max(50, int(config["persona_template_count"]))
    categories = [item["name"] for item in config["categories"]]
    age_bands = ["10대 후반", "20대 초반", "20대 후반", "30대", "40대", "50대 이상"]
    life_stages = [
        "고등학생", "대학생", "취업 준비", "사회초년생", "1인 자취",
        "신혼 생활", "영유아 양육", "학령기 자녀 양육", "이직 준비", "은퇴 후 생활",
    ]
    housing_types = ["가족과 거주", "기숙사", "원룸", "오피스텔", "아파트", "빌라", "공유주거"]
    occupations = [
        "학생", "사무직", "교대 근무", "프리랜서", "자영업", "현장직",
        "육아 중심", "재택근무", "구직 중", "은퇴 생활",
    ]
    short_needs = [
        "이사 준비", "자취 시작", "입학 준비", "새 직장 준비", "취미 시작",
        "연말 선물", "고장 난 물건 교체", "가족 부탁", "겨울 생활 준비",
        "급한 행사 준비", "생활비 마련을 위한 정리", "단순 호기심",
    ]
    budget_styles = ["절약형", "균형형", "품질 우선형", "한정 예산형", "상황별 유동형"]
    decision_styles = ["비교형", "신중형", "충동형", "질문 확인형", "경험 의존형", "후기 탐색형"]
    negotiation_styles = ["가격 제안 적극형", "필요할 때만 제안", "정가 수용형", "묶음 협상형", "거래 편의 우선형"]
    active_hours = ["출근 전", "점심시간", "저녁", "심야", "주말 중심", "시간 불규칙"]
    travel_ranges = ["동네 안", "인접 지역", "대중교통 30분", "광역 이동 가능", "택배 선호"]
    selling_styles = ["빠른 정리형", "설명 충실형", "가격 유지형", "협상 가능형", "가끔 판매", "묶음 정리형"]
    triggers = [
        "추천 피드 우연 노출", "친구의 부탁", "가족의 부탁", "선물 구매",
        "새 취미 체험", "제품 고장", "학교·직장 필요", "가격이 매우 저렴해 확인",
        "동네에서 가까워 확인", "일시적인 관심 변화", "단순 구경", "이사 중 예상 밖 필요",
    ]
    exceptions = [
        "주요 관심사와 무관한 상품도 선물 목적이면 오래 비교한다.",
        "절약을 중시하지만 희귀 취미 상품에는 높은 가격을 수용할 수 있다.",
        "충동적으로 조회해도 실제 구매 전에는 여러 번 다시 확인한다.",
        "이동 범위가 좁지만 꼭 필요한 물건이면 먼 지역도 고려한다.",
        "협상을 선호하지만 급한 구매에서는 제시 가격을 그대로 받아들인다.",
        "온라인 활동은 적지만 주말에 짧게 많은 상품을 탐색한다.",
        "평소 중고 구매가 적어도 고장 교체 상황에서는 즉시 탐색한다.",
        "관심 카테고리라도 판매글 신뢰가 낮으면 아무 행동 없이 떠난다.",
    ]

    templates: list[dict[str, Any]] = []
    for index in range(count):
        primary = [categories[index % len(categories)]]
        if index % 4 == 0:
            primary.append(categories[(index * 5 + 3) % len(categories)])
        secondary = [categories[(index * 7 + 2) % len(categories)]]
        if index % 3 == 0:
            secondary.append(categories[(index * 11 + 6) % len(categories)])
        short_category = categories[(index * 9 + 4) % len(categories)]
        templates.append(
            {
                "template_id": f"persona_template_{index + 1:03d}",
                "age_band": age_bands[index % len(age_bands)],
                "life_stage": life_stages[(index * 3) % len(life_stages)],
                "housing_type": housing_types[(index * 5) % len(housing_types)],
                "occupation_context": occupations[(index * 7) % len(occupations)],
                "primary_interests": list(dict.fromkeys(primary)),
                "secondary_interests": list(dict.fromkeys(secondary)),
                "current_short_need": short_needs[(index * 5 + 1) % len(short_needs)],
                "short_need_categories": [short_category],
                "budget_style": budget_styles[(index * 2) % len(budget_styles)],
                "decision_style": decision_styles[(index * 5) % len(decision_styles)],
                "negotiation_style": negotiation_styles[(index * 3) % len(negotiation_styles)],
                "active_hours": active_hours[(index * 5) % len(active_hours)],
                "travel_range": travel_ranges[(index * 3) % len(travel_ranges)],
                "selling_style": selling_styles[(index * 7) % len(selling_styles)],
                "outside_interest_trigger": triggers[(index * 7 + 2) % len(triggers)],
                "exception_note": exceptions[(index * 3 + 1) % len(exceptions)],
            }
        )

    return {
        "asset_type": "persona_templates",
        "version": config["llm"]["cache_version"],
        "source": "codex_curated_fallback",
        "templates": templates,
    }


def _fallback_listing_scenarios(config: dict[str, Any]) -> dict[str, Any]:
    count = max(60, int(config["listing_scenario_count"]))
    categories = [item["name"] for item in config["categories"]]
    seller_situations = [
        "집 안 물건을 정리하는 중", "새 제품을 선물 받아 기존 제품을 정리", "이사 전 부피를 줄이는 중",
        "취미가 바뀌어 사용 빈도가 낮아짐", "급한 생활비가 필요함", "가족 대신 판매를 맡음",
        "보관 공간이 부족함", "상위 모델로 교체함", "한동안 사용하지 않아 판매", "묶음 구매 후 남은 물건을 정리",
    ]
    reasons = [
        "사용 빈도 감소", "중복 보유", "공간 확보", "이사 정리", "취미 변경",
        "선물로 새 제품 수령", "급처 필요", "사이즈 불일치", "가족 부탁", "기기 교체",
    ]
    conditions = [
        "사용감이 거의 없음", "생활 사용감이 조금 있음", "기능은 정상이나 외관 흠집이 있음",
        "구성품 일부가 없음", "미개봉 상태", "오래됐지만 관리가 잘됨", "수리가 필요한 부분을 고지함",
        "사진으로 확인하기 어려운 작은 흔적이 있음",
    ]
    urgencies = ["여유 있게 판매", "적당한 기간 안에 판매", "이번 주 안에 정리", "당일 거래 선호", "가격보다 편한 거래 우선"]
    price_positions = ["bargain", "typical", "typical", "typical", "premium", "rare"]
    title_styles = ["담백한 사실형", "상태 강조형", "급처 강조형", "구성품 강조형", "짧은 구어체", "정중한 설명형"]
    description_styles = ["간결형", "상세형", "대화형", "사실 중심형", "서사형", "목록형"]
    title_patterns = [
        "{product} 판매합니다", "{condition} {product}", "{product} {price_note}",
        "{reason}으로 {product} 정리해요", "구성 좋은 {product}", "{region} 직거래 {product}",
    ]
    description_patterns = [
        "{reason} 때문에 판매합니다. {condition}. {usage_note} {trade_note}",
        "상품명: {product}\n상태: {condition}\n판매 이유: {reason}\n{trade_note}",
        "잘 사용하던 {product}입니다. {condition}. 필요한 분께 갔으면 합니다. {trade_note}",
        "{product}이고 {condition}. 사진에 보이는 구성으로 드립니다. {usage_note} {trade_note}",
        "구매 후 {usage_note} {reason}으로 내놓습니다. {condition}. {trade_note}",
        "{condition} / {usage_note} / {reason}. {trade_note}",
    ]
    winter_contexts = [
        "추운 날씨로 실내 생활이 늘어남", "연말 선물을 준비함", "겨울방학 동안 새 취미를 시작함",
        "겨울 스포츠를 준비함", "난방용품이 갑자기 필요해짐", "겨울 이사와 자취를 준비함",
        "계절 의류를 정리함", "실내 취미 시간이 늘어남", "연초 생활 정리를 시작함",
        "눈 오는 날 이동이 어려워 동네 거래를 선호함",
    ]
    cross_contexts = [
        "가족의 부탁으로 평소 보지 않던 카테고리를 탐색", "친구 선물로 관심 밖 상품을 비교",
        "새 직장에서 필요해 생소한 상품을 찾음", "이사 중 예상하지 못한 생활용품이 필요",
        "추천 피드에서 우연히 보고 호기심이 생김", "물건 고장으로 대체품을 급히 탐색",
        "새 취미를 시험해 보기 위해 저렴한 입문품을 찾음", "학교 행사 때문에 일회성 물품이 필요",
    ]
    rare_cases = [
        "시세보다 매우 비싼 희귀 구성", "시세보다 매우 낮은 급처 가격", "사진은 적지만 설명이 솔직한 판매글",
        "설명은 매우 길지만 핵심 정보가 분산됨", "계절과 반대되는 상품을 겨울에 정리",
        "짧은 기간 안에 가격을 여러 번 조정", "일반 상품처럼 보이지만 수집 가치가 있음",
    ]

    scenarios: list[dict[str, Any]] = []
    for index in range(count):
        if index < round(count * 0.57):
            context_type = "general"
        elif index < round(count * 0.74):
            context_type = "cross_category"
        elif index < round(count * 0.95):
            context_type = "seasonal"
        else:
            context_type = "rare"
        season_related = context_type == "seasonal"
        cross_context = cross_contexts[index % len(cross_contexts)] if context_type == "cross_category" else "해당 없음"
        rare_case = rare_cases[index % len(rare_cases)] if context_type == "rare" else "해당 없음"
        situation = winter_contexts[index % len(winter_contexts)] if season_related else seller_situations[(index * 3) % len(seller_situations)]
        scenarios.append(
            {
                "scenario_id": f"listing_scenario_{index + 1:03d}",
                "context_type": context_type,
                "category_hint": categories[(index * 5 + 2) % len(categories)],
                "seller_situation": situation,
                "selling_reason": reasons[(index * 7) % len(reasons)],
                "item_condition": conditions[(index * 5) % len(conditions)],
                "urgency": urgencies[(index * 3) % len(urgencies)],
                "price_position": price_positions[(index * 5) % len(price_positions)],
                "title_style": title_styles[(index * 5) % len(title_styles)],
                "description_style": description_styles[(index * 7) % len(description_styles)],
                "title_pattern": title_patterns[(index * 5) % len(title_patterns)],
                "description_pattern": description_patterns[(index * 7) % len(description_patterns)],
                "season_related": season_related,
                "cross_category_context": cross_context,
                "rare_case": rare_case,
            }
        )

    return {
        "asset_type": "listing_scenarios",
        "version": config["llm"]["cache_version"],
        "source": "codex_curated_fallback",
        "writing_styles": {
            "title_styles": title_styles,
            "description_styles": description_styles,
            "title_patterns": title_patterns,
            "description_patterns": description_patterns,
        },
        "scenarios": scenarios,
    }


def _refresh_with_openai(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    if not api_key or not model:
        raise RuntimeError("OPENAI_API_KEY와 OPENAI_MODEL이 모두 필요합니다.")

    client = OpenAI(api_key=api_key)
    persona_prompt = (project_root / "prompts/persona_prompt.txt").read_text(encoding="utf-8")
    listing_prompt = (project_root / "prompts/listing_scenario_prompt.txt").read_text(encoding="utf-8")

    persona_response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": f"서로 다른 템플릿 {config['persona_template_count']}개를 생성하라."},
        ],
        text_format=PersonaTemplateBatch,
    )
    listing_response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": listing_prompt},
            {"role": "user", "content": f"서로 다른 판매 상황 {config['listing_scenario_count']}개를 생성하라."},
        ],
        text_format=ListingScenarioBatch,
    )
    if persona_response.output_parsed is None or listing_response.output_parsed is None:
        raise RuntimeError("Structured Outputs 응답을 파싱하지 못했습니다.")

    personas = [item.model_dump() for item in persona_response.output_parsed.templates]
    scenarios = [item.model_dump() for item in listing_response.output_parsed.scenarios]
    if len(personas) < 50 or len(scenarios) < 60:
        raise RuntimeError("LLM 응답의 템플릿 수가 최소 요구량보다 적습니다.")

    for index, item in enumerate(personas, start=1):
        item["template_id"] = f"persona_template_{index:03d}"
    for index, item in enumerate(scenarios, start=1):
        item["scenario_id"] = f"listing_scenario_{index:03d}"

    persona_asset = {
        "asset_type": "persona_templates",
        "version": config["llm"]["cache_version"],
        "source": f"openai:{model}",
        "templates": personas,
    }
    listing_asset = {
        "asset_type": "listing_scenarios",
        "version": config["llm"]["cache_version"],
        "source": f"openai:{model}",
        "scenarios": scenarios,
    }
    return persona_asset, listing_asset


def ensure_llm_assets(
    config: dict[str, Any],
    project_root: Path,
    refresh_requested: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """캐시를 준비하고 자산 및 캐시 메타데이터를 반환한다."""

    load_dotenv(project_root / ".env")
    generated_dir = project_root / "generated"
    cache_dir = generated_dir / "llm_cache"
    persona_path = generated_dir / "persona_templates.json"
    listing_path = generated_dir / "listing_scenarios.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    should_refresh = refresh_requested or bool(config["llm"].get("refresh_with_api"))
    api_status = "not_requested"
    if should_refresh:
        try:
            persona_asset, listing_asset = _refresh_with_openai(config, project_root)
            api_status = "refreshed"
        except Exception as exc:  # API 실패는 캐시 파이프라인을 막지 않는다.
            api_status = f"fallback_after_api_error: {type(exc).__name__}: {exc}"
            persona_asset = _read_json(persona_path) if persona_path.exists() else _fallback_personas(config)
            listing_asset = _read_json(listing_path) if listing_path.exists() else _fallback_listing_scenarios(config)
    else:
        persona_asset = _read_json(persona_path) if persona_path.exists() else _fallback_personas(config)
        listing_asset = _read_json(listing_path) if listing_path.exists() else _fallback_listing_scenarios(config)

    if len(persona_asset.get("templates", [])) < 50:
        persona_asset = _fallback_personas(config)
        api_status = "fallback_invalid_persona_cache"
    if len(listing_asset.get("scenarios", [])) < 60:
        listing_asset = _fallback_listing_scenarios(config)
        api_status = "fallback_invalid_listing_cache"

    _write_json(persona_path, persona_asset)
    _write_json(listing_path, listing_asset)
    cache_version = config["llm"]["cache_version"]
    _write_json(cache_dir / f"persona_templates-{cache_version}.json", persona_asset)
    _write_json(cache_dir / f"listing_scenarios-{cache_version}.json", listing_asset)

    digest = hashlib.sha256(
        json.dumps([persona_asset, listing_asset], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    metadata = {
        "cache_version": cache_version,
        "cache_digest": digest,
        "persona_source": persona_asset.get("source", "unknown"),
        "listing_source": listing_asset.get("source", "unknown"),
        "api_status": api_status,
    }
    return persona_asset, listing_asset, metadata

