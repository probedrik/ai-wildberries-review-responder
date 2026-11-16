from __future__ import annotations

from src.application.ports import AppLogger, ReplyGenerator, ReviewFetcher, ReviewPublisher


def respond_on_reviews(
    *,
    review_fetcher: ReviewFetcher,
    reply_generator: ReplyGenerator,
    review_publisher: ReviewPublisher,
    logger: AppLogger,
    dry_run: bool = False,
) -> None:
    """Загрузить неотвеченные отзывы, сгенерировать ответы и опубликовать их.

    ``dry_run=True`` печатает готовые ответы в лог и НЕ публикует ничего:
    так прогон проверяется на живых отзывах до боевого запуска.
    Пустой ответ (модель не справилась или не прошла валидацию) означает пропуск
    отзыва — публиковать заглушку нельзя.
    """

    reviews = review_fetcher.fetch_reviews()
    logger.info("Fetched {} pending reviews", len(reviews))

    published = 0
    skipped = 0

    for review in reviews:
        review_summary = review.summary
        logger.info(
            f"Preparing reply for review {review_summary}",
        )

        reply = reply_generator.generate(review)

        if not reply or not reply.strip():
            skipped += 1
            logger.warning("No usable reply for review {} — skipped", review_summary)
            continue

        if dry_run:
            logger.info("[DRY-RUN] Would publish for review {}: {}", review_summary, reply)
            published += 1
            continue

        logger.info("Generated reply {}. Publishing...", reply)
        review_publisher.publish_reply(review.id, reply)
        published += 1
        logger.info("Published reply for review {}", review_summary)

    logger.info(
        "Finished: {} prepared, {} skipped (dry_run={})",
        published,
        skipped,
        dry_run,
    )
