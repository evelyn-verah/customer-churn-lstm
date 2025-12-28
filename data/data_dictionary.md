# Data Dictionary – Mobile Money Logs

| Column       | Type    | Description                                        |
|--------------|---------|----------------------------------------------------|
| user_id      | int     | Unique customer identifier                         |
| day          | int     | Day index (1–90) within the observation window    |
| txn_count    | int     | Number of transactions on that day                |
| cashin       | float   | Total cash-in amount for the day                  |
| cashout      | float   | Total cash-out amount for the day                 |
| failed_login | int(0/1)| Indicator for failed login attempt                |
| pin_reset    | int(0/1)| Indicator for PIN reset event                     |
| churned      | int(0/1)| Target label: 1 = churned, 0 = retained           |
