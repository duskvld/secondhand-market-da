-- 기존 데이터베이스가 있으면 삭제
DROP DATABASE IF EXISTS carrot_chat;

-- 데이터베이스 생성
CREATE DATABASE carrot_chat
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE carrot_chat;


-- =====================================================
-- 1. 사용자 테이블
-- =====================================================
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    signup_at DATETIME NOT NULL,
    region_id VARCHAR(20) NOT NULL,
    manner_score DECIMAL(4,1) NOT NULL DEFAULT 36.5,
    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
    last_active_at DATETIME NOT NULL,

    CHECK (manner_score BETWEEN 0.0 AND 100.0),
    CHECK (is_verified IN (0, 1)),
    CHECK (last_active_at >= signup_at)
) ENGINE = InnoDB;


-- =====================================================
-- 2. 판매글 테이블
-- =====================================================
CREATE TABLE listings (
    listing_id INT PRIMARY KEY,
    seller_id INT NOT NULL,
    category VARCHAR(50) NOT NULL,
    title VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    photo_count SMALLINT NOT NULL DEFAULT 0,
    price DECIMAL(12,0) NOT NULL,
    is_negotiable BOOLEAN NOT NULL DEFAULT FALSE,
    region_id VARCHAR(20) NOT NULL,
    created_at DATETIME NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT '판매중',

    FOREIGN KEY (seller_id)
        REFERENCES users(user_id),

    CHECK (photo_count >= 0),
    CHECK (price >= 0),
    CHECK (is_negotiable IN (0, 1)),
    CHECK (status IN ('판매중', '예약중', '판매완료', '삭제'))
) ENGINE = InnoDB;


-- =====================================================
-- 3. 판매글 조회 기록 테이블
-- =====================================================
CREATE TABLE product_views (
    view_id INT PRIMARY KEY,
    user_id INT NOT NULL,
    listing_id INT NOT NULL,
    viewed_at DATETIME NOT NULL,
    source VARCHAR(30) NOT NULL,

    FOREIGN KEY (user_id)
        REFERENCES users(user_id),

    FOREIGN KEY (listing_id)
        REFERENCES listings(listing_id),

    CHECK (
        source IN (
            'search',
            'recommendation',
            'category',
            'favorite_list',
            'seller_profile',
            'other'
        )
    )
) ENGINE = InnoDB;


-- =====================================================
-- 4. 찜 기록 테이블
-- =====================================================
CREATE TABLE favorites (
    favorite_id INT PRIMARY KEY,
    user_id INT NOT NULL,
    listing_id INT NOT NULL,
    added_view_id INT NOT NULL,
    favorited_at DATETIME NOT NULL,
    removed_view_id INT NULL,
    removed_at DATETIME NULL,

    FOREIGN KEY (user_id)
        REFERENCES users(user_id),

    FOREIGN KEY (listing_id)
        REFERENCES listings(listing_id),

    FOREIGN KEY (added_view_id)
        REFERENCES product_views(view_id),

    FOREIGN KEY (removed_view_id)
        REFERENCES product_views(view_id),

    CHECK (
        (removed_view_id IS NULL AND removed_at IS NULL)
        OR
        (removed_view_id IS NOT NULL AND removed_at IS NOT NULL)
    ),

    CHECK (
        removed_at IS NULL
        OR removed_at >= favorited_at
    )
) ENGINE = InnoDB;


-- =====================================================
-- 5. 채팅방 테이블
-- =====================================================
CREATE TABLE chatrooms (
    chatroom_id INT PRIMARY KEY,
    listing_id INT NOT NULL,
    buyer_id INT NOT NULL,
    seller_id INT NOT NULL,
    created_at DATETIME NOT NULL,
    first_message_at DATETIME NULL,
    seller_response_at DATETIME NULL,
    message_count INT NOT NULL DEFAULT 0,
    chat_status VARCHAR(20) NOT NULL DEFAULT '진행중',

    FOREIGN KEY (listing_id)
        REFERENCES listings(listing_id),

    FOREIGN KEY (buyer_id)
        REFERENCES users(user_id),

    FOREIGN KEY (seller_id)
        REFERENCES users(user_id),

    UNIQUE (listing_id, buyer_id),

    CHECK (buyer_id <> seller_id),
    CHECK (message_count >= 0),

    CHECK (
        first_message_at IS NULL
        OR first_message_at >= created_at
    ),

    CHECK (
        seller_response_at IS NULL
        OR (
            first_message_at IS NOT NULL
            AND seller_response_at >= first_message_at
        )
    ),

    CHECK (
        (message_count = 0 AND first_message_at IS NULL)
        OR
        (message_count >= 1 AND first_message_at IS NOT NULL)
    ),

    CHECK (
        seller_response_at IS NULL
        OR message_count >= 2
    ),

    CHECK (
        chat_status IN ('진행중', '종료', '차단')
    )
) ENGINE = InnoDB;


-- =====================================================
-- 6. 거래 테이블
-- =====================================================
CREATE TABLE transactions (
    transaction_id INT PRIMARY KEY,
    listing_id INT NOT NULL,
    chatroom_id INT NOT NULL UNIQUE,
    buyer_id INT NOT NULL,
    seller_id INT NOT NULL,
    transaction_status VARCHAR(20) NOT NULL DEFAULT '거래중',
    started_at DATETIME NOT NULL,
    completed_at DATETIME NULL,
    ended_at DATETIME NULL,

    FOREIGN KEY (listing_id)
        REFERENCES listings(listing_id),

    FOREIGN KEY (chatroom_id)
        REFERENCES chatrooms(chatroom_id),

    FOREIGN KEY (buyer_id)
        REFERENCES users(user_id),

    FOREIGN KEY (seller_id)
        REFERENCES users(user_id),

    CHECK (buyer_id <> seller_id),

    CHECK (
        transaction_status IN (
            '거래중',
            '거래완료',
            '거래종료'
        )
    ),

    CHECK (
        (
            transaction_status = '거래중'
            AND completed_at IS NULL
            AND ended_at IS NULL
        )
        OR
        (
            transaction_status = '거래완료'
            AND completed_at IS NOT NULL
            AND ended_at IS NULL
        )
        OR
        (
            transaction_status = '거래종료'
            AND completed_at IS NULL
            AND ended_at IS NOT NULL
        )
    ),

    CHECK (
        completed_at IS NULL
        OR completed_at >= started_at
    ),

    CHECK (
        ended_at IS NULL
        OR ended_at >= started_at
    )
) ENGINE = InnoDB;


-- =====================================================
-- 7. 가격 변경 이력 테이블
-- =====================================================
CREATE TABLE price_history (
    price_history_id INT PRIMARY KEY,
    listing_id INT NOT NULL,
    price DECIMAL(12,0) NOT NULL,
    record_type VARCHAR(20) NOT NULL,
    recorded_at DATETIME NOT NULL,

    FOREIGN KEY (listing_id)
        REFERENCES listings(listing_id),

    UNIQUE (listing_id, recorded_at),

    CHECK (price >= 0),

    CHECK (
        record_type IN ('initial', 'changed')
    )
) ENGINE = InnoDB;