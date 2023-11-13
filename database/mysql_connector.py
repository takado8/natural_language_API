import mysql.connector


class MySQLConnector:
    def __init__(self):
        mydb = mysql.connector.connect(
            host="localhost",
            user="yourusername",
            password="yourpassword"
        )

        print(mydb)

    def run(self):
        pass