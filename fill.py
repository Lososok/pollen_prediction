from datetime import datetime
from datetime import timedelta

# today = datetime.today()
start = datetime(2019, 1, 1)
with open('pollen.csv', "+a") as data:
    for i in range(365 * 2):
        data.write(
            str(start + timedelta(days=i)).split()[0] + ",0\n"
        )