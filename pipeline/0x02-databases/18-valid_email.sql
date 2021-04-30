-- SQL script that creates a trigger that resets the attribute valid_email only when the email has been changed.
-- Context: Nothing related to MySQL, but perfect for user email validation - distribute the logic to the database itself!
DROP TRIGGER IF EXISTS users.reset_valid_email;
DELIMITER $$
CREATE TRIGGER reset_valid_email
BEFORE UPDATE
ON users
FOR EACH ROW
BEGIN
	IF STRCMP(OLD.email, NEW.email) != 0 THEN
		SET NEW.valid_email = 0;
	END IF;
END;$$
DELIMITER ;
