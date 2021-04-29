-- Script that lists all shows from hbtn_0d_tvshows_rate by their rating.
-- Import the database hbtn_0d_tvshows_rate to your MySQL server
-- 		Each record displays: tv_shows.title - rating sum
--	    Results are sorted in descending order by the rating
-- 		This script uses only one SELECT statement
-- 		The database name is passed as an argument of the mysql command
SELECT tv_shows.title, SUM(tv_show_ratings.rate) AS rating
FROM tv_shows
INNER JOIN tv_show_ratings
ON tv_shows.id = tv_show_ratings.show_id
GROUP BY tv_shows.title
ORDER BY rating DESC;
