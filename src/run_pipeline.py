"""본 데이터 생성·검증·선택적 MySQL 적재 파이프라인 진입점."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llm_assets import ensure_llm_assets
from load_to_mysql import load_to_mysql
from simulate_dataset import simulate_dataset, write_simulation_outputs
from validate_dataset import validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config/generation_config.json"


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _write_manifest(
    config: dict[str, Any],
    report: dict[str, Any],
    simulation_metadata: dict[str, Any],
    llm_metadata: dict[str, Any],
    mysql_result: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "observation_period": {
            "start": config["observation_start"],
            "end": config["observation_end"],
        },
        "target_season": config["target_season"],
        "master_seed": config["master_seed"],
        "sub_seeds": config["seeds"],
        "target_scale": {
            "users": config["user_count"],
            "listings": config["listing_count"],
            "product_views": config["view_count_range"],
            "favorites": [4000, 8000],
            "chatrooms": [1000, 3000],
            "transactions": [300, 1000],
        },
        "actual_row_counts": report["row_counts"],
        "seasonality_strength": simulation_metadata["seasonality_strength"],
        "season_related_listing_ratio": simulation_metadata["season_related_listing_ratio"],
        "configured_anomaly_rate": simulation_metadata["configured_anomaly_rate"],
        "actual_unique_anomaly_row_ratio": report["anomalies"]["unique_row_ratio_of_all_rows"],
        "llm_cache_version": llm_metadata["cache_version"],
        "llm_cache_digest": llm_metadata["cache_digest"],
        "llm_asset_sources": {
            "personas": llm_metadata["persona_source"],
            "listing_scenarios": llm_metadata["listing_source"],
            "api_status": llm_metadata["api_status"],
        },
        "generation_config_version": config["config_version"],
        "integrity_validation_passed": report["passed_integrity"],
        "mysql_load_success": bool(mysql_result.get("success")),
        "mysql_load_result": mysql_result,
        "assumptions": report["assumptions"],
    }
    path = PROJECT_ROOT / "generated/run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_pipeline(refresh_llm: bool = False, skip_mysql: bool = False) -> int:
    """캐시 준비부터 안전 적재까지 실행하고 프로세스 종료 코드를 반환한다."""

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = _load_config()
    for directory in [PROJECT_ROOT / "generated/csv", PROJECT_ROOT / "generated/llm_cache", PROJECT_ROOT / "reports"]:
        directory.mkdir(parents=True, exist_ok=True)

    print("[1/5] LLM 의미 자산 준비")
    refresh_requested = refresh_llm or _env_true("REFRESH_LLM_ASSETS")
    persona_asset, listing_asset, llm_metadata = ensure_llm_assets(
        config, PROJECT_ROOT, refresh_requested=refresh_requested,
    )
    print(f"  persona templates: {len(persona_asset['templates'])}")
    print(f"  listing scenarios: {len(listing_asset['scenarios'])}")

    print("[2/5] 7개 테이블 본 데이터 생성")
    simulation = simulate_dataset(config, persona_asset, listing_asset)
    write_simulation_outputs(simulation, PROJECT_ROOT)

    print("[3/5] 무결성 및 분석 분포 검증")
    report = validate_dataset(config, PROJECT_ROOT, simulation.metadata)
    if report["passed_integrity"]:
        print("  치명적 무결성 오류: 0")
    else:
        print(f"  치명적 무결성 오류: {report['fatal_error_count']}")

    print("[4/5] MySQL 안전 적재 확인")
    load_enabled = bool(config["mysql"]["load_enabled"]) and not skip_mysql
    mysql_result = load_to_mysql(PROJECT_ROOT, load_enabled, report["passed_integrity"])
    print(f"  {mysql_result['status']}: {mysql_result['message']}")

    print("[5/5] 실행 manifest 기록")
    _write_manifest(config, report, simulation.metadata, llm_metadata, mysql_result)

    print("\n생성 행 수")
    for table, count in report["row_counts"].items():
        print(f"  {table}: {count:,}")
    print(f"계절 관련 판매글 비율: {report['seasonality']['season_related_listing_ratio']:.2%}")
    print(
        "이상치: "
        f"{report['anomalies']['unique_row_count']:,}개 고유 행 "
        f"({report['anomalies']['unique_row_ratio_of_all_rows']:.2%})"
    )
    print(f"검증 통과: {report['passed_integrity']}")
    print(f"MySQL 적재 성공: {mysql_result.get('success', False)}")
    return 0 if report["passed_integrity"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="중고거래 합성 데이터 파이프라인")
    parser.add_argument("--refresh-llm", action="store_true", help=".env의 OpenAI 설정으로 의미 자산을 다시 생성")
    parser.add_argument("--skip-mysql", action="store_true", help="MySQL 환경변수가 있어도 이번 실행에서는 적재하지 않음")
    args = parser.parse_args()
    raise SystemExit(run_pipeline(refresh_llm=args.refresh_llm, skip_mysql=args.skip_mysql))


if __name__ == "__main__":
    main()
