-- Script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending).
-- Import in hbtn_0c_0 database the temperatures.sql table
SELECT city, AVG(`value`) AS avg_temp
FROM temperatures
GROUP BY city
ORDER BY avg_temp DESC;
