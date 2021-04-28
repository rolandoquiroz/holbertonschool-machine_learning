-- Script that lists all shows contained in hbtn_0d_tvshows that have at least one genre linked.
-- 	Each record should display: tv_shows.title - tv_show_genres.genre_id
-- 	Results must be sorted in ascending order by tv_shows.title and tv_show_genres.genre_id
-- 	This script uses only one SELECT statement
-- 	The database name is passed as an argument of the mysql command
-- Note: Columns containing NULL do not match any values when you are creating an inner join
-- and are therefore excluded from the result set. Null values do not match other null values.
SELECT tv_shows.title, tv_show_genres.genre_id
FROM tv_shows
INNER JOIN tv_show_genres
ON tv_shows.id = tv_show_genres.show_id
-- WHERE tv_show_genres.genre_id IS NOT NULL
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;
