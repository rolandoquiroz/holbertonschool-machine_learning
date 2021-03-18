-- SQL script that ranks country origins of bands, ordered by the number of (non-unique) fans
-- Requirements:
--  Import the table: metal_bands.sql
--  Column names must be: origin and nb_fans
--  Your script can be executed on any database
SELECT origin, sum(fans) AS nb_fans FROM metal_bands GROUP BY origin ORDER BY nb_fans DESC;
