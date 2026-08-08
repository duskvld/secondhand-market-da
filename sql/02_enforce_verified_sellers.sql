-- 미인증 사용자의 판매글 등록과 판매자 변경을 DB 수준에서 차단한다.
-- 기존 판매자가 인증 상태를 해제하여 규칙을 우회하는 것도 차단한다.

USE carrot_chat;

DROP TRIGGER IF EXISTS trg_listings_verified_seller_insert;
DROP TRIGGER IF EXISTS trg_listings_verified_seller_update;
DROP TRIGGER IF EXISTS trg_users_keep_seller_verified;

DELIMITER //

CREATE TRIGGER trg_listings_verified_seller_insert
BEFORE INSERT ON listings
FOR EACH ROW
BEGIN
    IF COALESCE(
        (SELECT is_verified FROM users WHERE user_id = NEW.seller_id),
        FALSE
    ) <> TRUE THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = '판매글 작성자는 동네 인증이 필요합니다.';
    END IF;
END//

CREATE TRIGGER trg_listings_verified_seller_update
BEFORE UPDATE ON listings
FOR EACH ROW
BEGIN
    IF COALESCE(
        (SELECT is_verified FROM users WHERE user_id = NEW.seller_id),
        FALSE
    ) <> TRUE THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = '판매글 작성자는 동네 인증이 필요합니다.';
    END IF;
END//

CREATE TRIGGER trg_users_keep_seller_verified
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF OLD.is_verified = TRUE
       AND NEW.is_verified = FALSE
       AND EXISTS (
           SELECT 1
           FROM listings
           WHERE seller_id = OLD.user_id
       ) THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = '판매글이 있는 사용자는 동네 인증을 해제할 수 없습니다.';
    END IF;
END//

DELIMITER ;
