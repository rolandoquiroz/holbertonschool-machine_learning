-- Script that lists all genres in the database hbtn_0d_tvshows_rate by their rating.
-- Import the database hbtn_0d_tvshows_rate to your MySQL server
--      Each record displays: tv_genres.name - rating sum
--      Results are sorted in descending order by their rating
--      This script uses only one SELECT statement
--      The database name is passed as an argument of the mysql command
SELECT `name`, SUM(rate) AS rating
FROM tv_show_ratings
NATURAL JOIN tv_show_genres 
INNER JOIN tv_genres
ON genre_id = tv_genres.id
GROUP BY `name`
ORDER BY rating DESC;
