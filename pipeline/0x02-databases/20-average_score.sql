-- SQL script that creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
-- Requirements:
--  Procedure ComputeAverageScoreForUser is taking 1 input:
--      user_id, a users.id value (you can assume user_id is linked to an existing users)
DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;
DELIMITER $$
CREATE PROCEDURE ComputeAverageScoreForUser(IN new_user_id INT)
BEGIN
	UPDATE users
	SET average_score = (SELECT AVG(score)
	                     FROM corrections
						 WHERE user_id=new_user_id)
			             WHERE id=new_user_id;
END $$
DELIMITER ;