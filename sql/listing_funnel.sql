USE carrot_chat;

CREATE OR REPLACE VIEW vw_listing_funnel AS

/* 판매글별 조회 수 계산 */
WITH view_agg AS (
    SELECT
        listing_id,
        COUNT(*) AS view_count,
        COUNT(DISTINCT user_id) AS unique_viewer_count
    FROM product_views
    GROUP BY listing_id
),

/* 판매글별 찜 수 계산 */
favorite_agg AS (
    SELECT
        listing_id,
        COUNT(*) AS favorite_event_count,
        COUNT(DISTINCT user_id) AS unique_favoriter_count,

        SUM(
            CASE
                WHEN removed_at IS NOT NULL THEN 1
                ELSE 0
            END
        ) AS favorite_remove_count
    FROM favorites
    GROUP BY listing_id
),

/* 판매글별 실제 채팅 시작 수 계산 */
chat_agg AS (
    SELECT
        listing_id,

        SUM(
            CASE
                WHEN first_message_at IS NOT NULL THEN 1
                ELSE 0
            END
        ) AS chat_started_count,

        COUNT(
            DISTINCT CASE
                WHEN first_message_at IS NOT NULL THEN buyer_id
            END
        ) AS unique_chat_buyer_count
    FROM chatrooms
    GROUP BY listing_id
)

/* 위에서 계산한 결과를 판매글에 붙이기 */
SELECT
    l.listing_id,
    l.seller_id,
    l.category,
    l.price AS current_price,
    l.photo_count,

    /* 판매글 설명 글자 수 */
    CHAR_LENGTH(COALESCE(l.description, ''))
        AS description_length,

    l.is_negotiable,
    l.region_id,
    l.created_at,
    l.status,

    /* 판매자 정보 */
    u.manner_score AS seller_manner_score,
    u.is_verified AS seller_is_verified,

    /* 조회 정보 */
    COALESCE(v.view_count, 0)
        AS view_count,

    COALESCE(v.unique_viewer_count, 0)
        AS unique_viewer_count,

    /* 찜 정보 */
    COALESCE(f.favorite_event_count, 0)
        AS favorite_event_count,

    COALESCE(f.unique_favoriter_count, 0)
        AS unique_favoriter_count,

    COALESCE(f.favorite_remove_count, 0)
        AS favorite_remove_count,

    /* 채팅 정보 */
    COALESCE(c.chat_started_count, 0)
        AS chat_started_count,

    COALESCE(c.unique_chat_buyer_count, 0)
        AS unique_chat_buyer_count

FROM listings l

LEFT JOIN users u
    ON l.seller_id = u.user_id

LEFT JOIN view_agg v
    ON l.listing_id = v.listing_id

LEFT JOIN favorite_agg f
    ON l.listing_id = f.listing_id

LEFT JOIN chat_agg c
    ON l.listing_id = c.listing_id;