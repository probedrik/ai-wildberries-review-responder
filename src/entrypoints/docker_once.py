from __future__ import annotations

import sys
from typing import cast

from src.application.ports import AppLogger
from src.application.respond_on_reviews import respond_on_reviews
from src.infra.clients import WildberriesClient
from src.infra.config.settings import Settings
from src.infra.llm.base import make_llm_client
from src.infra.logger import init_logger
from src.infra.prompt import PromptBuilder
from src.infra.reply_generator import LLMReplyGenerator


def run_once(dry_run: bool = False) -> None:
    settings = Settings()
    logger = cast(AppLogger, init_logger())

    wildberries_client = WildberriesClient(settings)
    llm_client = make_llm_client(settings, logger=logger)
    prompt_builder = PromptBuilder(template=settings.llm.prompt_template)
    reply_generator = LLMReplyGenerator(prompt_builder=prompt_builder, llm_client=llm_client)

    respond_on_reviews(
        review_fetcher=wildberries_client,
        reply_generator=reply_generator,
        review_publisher=wildberries_client,
        logger=logger,
        dry_run=dry_run,
    )


def main() -> None:
    import os

    dry_run = "--dry-run" in sys.argv or os.getenv("DRY_RUN", "").strip().lower() in {"1", "true", "yes"}
    if dry_run:
        print("DRY-RUN: ответы генерируются, но НЕ публикуются")
    run_once(dry_run=dry_run)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
