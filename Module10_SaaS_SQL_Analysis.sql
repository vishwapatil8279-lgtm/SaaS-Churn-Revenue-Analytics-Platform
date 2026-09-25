CREATE DATABASE saas_analytics;

USE saas_analytics;

CREATE TABLE customers (
    CustomerID VARCHAR(50) PRIMARY KEY,
    CompanyName VARCHAR(255),
    Industry VARCHAR(100),
    Country VARCHAR(100),
    City VARCHAR(100),
    EmployeeCount INT,
    SignupDate DATE,
    AcquisitionChannel VARCHAR(100)
);

CREATE TABLE subscriptions (
    SubscriptionID INT AUTO_INCREMENT PRIMARY KEY,
    CustomerID VARCHAR(50),
    MRR DECIMAL(12,2),
    Seats INT,
    StartDate DATE,
    Status VARCHAR(50)
);

CREATE TABLE usag (
    UsageID INT AUTO_INCREMENT PRIMARY KEY,
    CustomerID VARCHAR(50),
    Month DATE,
    Logins INT,
    ActiveUsers INT,
    APICalls INT,
    SessionMinutes INT
);

CREATE TABLE tickets (
    TicketID INT AUTO_INCREMENT PRIMARY KEY,
    CustomerID VARCHAR(50),
    OpenedDate DATE,
    ResolutionHours DECIMAL(10,2)
);

SHOW TABLES;
SELECT COUNT(*) AS CustomerRows
FROM customers;

SELECT COUNT(*) AS SubscriptionRows
FROM subscriptions;

SELECT COUNT(*) AS UsagRows
FROM usag;

SELECT COUNT(*) AS TicketRows
FROM tickets;

SELECT *
FROM customers
LIMIT 5;

SELECT *
FROM subscriptions
LIMIT 5;

SELECT *
FROM usag
LIMIT 5;

SELECT *
FROM tickets
LIMIT 5;

-- Query 1: JOIN

SELECT
    c.CustomerID,
    c.CompanyName,
    c.Industry,
    s.MRR,
    s.Seats,
    s.Status
FROM customers AS c
INNER JOIN subscriptions AS s
    ON c.CustomerID = s.CustomerID
ORDER BY s.MRR DESC;

-- Query 2: GROUP BY

SELECT
    CustomerID,
    SUM(MRR) AS TotalMRR,
    SUM(Seats) AS TotalSeats
FROM subscriptions
GROUP BY CustomerID
ORDER BY TotalMRR DESC;

-- Query 3: GROUP BY + HAVING

SELECT
    CustomerID,
    SUM(MRR) AS TotalMRR
FROM subscriptions
GROUP BY CustomerID
HAVING SUM(MRR) > 1000
ORDER BY TotalMRR DESC;

-- Query 4: CASE

SELECT
    CustomerID,
    SUM(MRR) AS TotalMRR,
    CASE
        WHEN SUM(MRR) >= 2000 THEN 'High Value'
        WHEN SUM(MRR) >= 1000 THEN 'Medium Value'
        ELSE 'Low Value'
    END AS CustomerValueSegment
FROM subscriptions
GROUP BY CustomerID
ORDER BY TotalMRR DESC;

-- Query 5: Subquery

SELECT
    CustomerID,
    SUM(MRR) AS TotalMRR
FROM subscriptions
GROUP BY CustomerID
HAVING SUM(MRR) > (
    SELECT AVG(CustomerMRR)
    FROM (
        SELECT
            CustomerID,
            SUM(MRR) AS CustomerMRR
        FROM subscriptions
        GROUP BY CustomerID
    ) AS customer_totals
)
ORDER BY TotalMRR DESC;

-- Query 6: CTE

WITH customer_usage AS (
    SELECT
        CustomerID,
        SUM(Logins) AS TotalLogins,
        SUM(APICalls) AS TotalAPICalls
    FROM usag
    GROUP BY CustomerID
),

customer_revenue AS (
    SELECT
        CustomerID,
        SUM(MRR) AS TotalMRR
    FROM subscriptions
    GROUP BY CustomerID
)

SELECT
    u.CustomerID,
    u.TotalLogins,
    u.TotalAPICalls,
    r.TotalMRR
FROM customer_usage AS u
LEFT JOIN customer_revenue AS r
    ON u.CustomerID = r.CustomerID
WHERE u.TotalLogins > 100
ORDER BY r.TotalMRR DESC;

-- Query 7: Window Function

WITH customer_revenue AS (
    SELECT
        CustomerID,
        SUM(MRR) AS TotalMRR
    FROM subscriptions
    GROUP BY CustomerID
)

SELECT
    CustomerID,
    TotalMRR,
    RANK() OVER (
        ORDER BY TotalMRR DESC
    ) AS MRRRank
FROM customer_revenue
ORDER BY MRRRank;

-- Query 8: Orphan records

SELECT
    s.CustomerID,
    s.MRR,
    s.Status
FROM subscriptions AS s
LEFT JOIN customers AS c
    ON s.CustomerID = c.CustomerID
WHERE c.CustomerID IS NULL;

SELECT
    COUNT(*) AS OrphanSubscriptionRecords
FROM subscriptions AS s
LEFT JOIN customers AS c
    ON s.CustomerID = c.CustomerID
WHERE c.CustomerID IS NULL;

-- Bonus Query: Churn by country

SELECT
    c.Country,
    COUNT(DISTINCT s.CustomerID) AS Customers,
    COUNT(
        DISTINCT CASE
            WHEN s.Status = 'Churned'
            THEN s.CustomerID
        END
    ) AS ChurnedCustomers
FROM customers AS c
LEFT JOIN subscriptions AS s
    ON c.CustomerID = s.CustomerID
GROUP BY c.Country
ORDER BY ChurnedCustomers DESC;

-- Bonus Query: Usage vs MRR

WITH usage_summary AS (
    SELECT
        CustomerID,
        SUM(Logins) AS TotalLogins,
        SUM(SessionMinutes) AS TotalSessionMinutes
    FROM usag
    GROUP BY CustomerID
),

revenue_summary AS (
    SELECT
        CustomerID,
        SUM(MRR) AS TotalMRR
    FROM subscriptions
    GROUP BY CustomerID
)

SELECT
    u.CustomerID,
    u.TotalLogins,
    u.TotalSessionMinutes,
    r.TotalMRR
FROM usage_summary AS u
LEFT JOIN revenue_summary AS r
    ON u.CustomerID = r.CustomerID
ORDER BY r.TotalMRR DESC;

