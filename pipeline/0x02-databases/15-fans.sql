-- SQL script that ranks country origins of bands, ordered by the number of (non-unique) fans
-- Requirements:
--      Import table metal_bands.sql
--      Column names are: origin and nb_fans
--      This script can be executed on any database
-- Context: Calculate/compute something is always power intensive… better to distribute the load!
SELECT origin, SUM(fans) AS nb_fans
FROM metal_bands
GROUP BY origin
HAVING nb_fans > 1
ORDER BY nb_fans DESC;
