-- Script that lists all genres in the database hbtn_0d_tvshows_rate by their rating.
-- Each record displays: tv_genres.name - rating sum
-- Results are sorted in descending order by their rating
-- This script uses only one SELECT statement
-- The database name is passed as an argument of the mysql command
SELECT tv_genres.name, SUM(tv_show_ratings.rate) AS rating FROM tv_genres
	JOIN tv_show_genres
	ON tv_genres.id = tv_show_genres.genre_id
	JOIN tv_show_ratings
	ON  tv_show_genres.show_id = tv_show_ratings.show_id
	GROUP BY tv_genres.name
	ORDER BY rating DESC;