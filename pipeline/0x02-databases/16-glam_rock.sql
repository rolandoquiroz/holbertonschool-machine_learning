-- SQL script that lists all bands with Glam rock as their main style, ranked by their longevity
-- Requirements:
--      Import table metal_bands.sql
--      Column names are: band_name and lifespan (in years)
--      lifespan is computed from attributes formed and split
--      This script can be executed on any database
SELECT band_name, IF (split IS NULL, (YEAR(NOW()) - formed), (split - formed)) AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam Rock%'
ORDER BY lifespan DESC;
