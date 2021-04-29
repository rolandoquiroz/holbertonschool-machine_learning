-- Lists all shows contained in hbtn_0d_tvshows without a genre linked.
-- Import in hbtn_0c_0 database the temperatures.sql table
-- 	Each record displays: tv_shows.title - tv_show_genres.genre_id
-- 	Results are sorted in ascending order by tv_shows.title and tv_show_genres.genre_id
-- 	This script uses only one SELECT statement
-- 	The database name is passed as an argument of the mysql command
-- MINUS emulated with LEFT JOIN
SELECT tv_shows.title, tv_show_genres.genre_id
FROM tv_shows
LEFT JOIN tv_show_genres
ON tv_shows.id = tv_show_genres.show_id
WHERE tv_show_genres.genre_id IS NULL
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;
