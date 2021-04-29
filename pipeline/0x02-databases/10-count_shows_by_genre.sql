-- Lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.
-- Import in hbtn_0c_0 database the temperatures.sql table
-- 		Each record displays: <TV Show genre> - <Number of shows linked to this genre>
-- 		First column is called genre
-- 		Second column is called number_of_shows
-- 		Don’t display a genre that doesn’t have any shows linked
-- 		Results are sorted in descending order by the number of shows linked
-- 		This script uses only one SELECT statement
-- 		The database name is passed as an argument of the mysql command
SELECT tv_genres.name AS genre, COUNT(tv_show_genres.show_id) AS number_of_shows
FROM tv_genres
INNER JOIN tv_show_genres
ON tv_genres.id = tv_show_genres.genre_id
GROUP BY genre
ORDER BY number_of_shows DESC;
