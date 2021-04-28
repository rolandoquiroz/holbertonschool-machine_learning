-- Script that computes the score average of all records in the table second_table in your MySQL server.
--  The result column name is average
--  The database name is passed as an argument of the mysql command
SELECT ROUND(AVG(score), 2) AS average
FROM second_table;
