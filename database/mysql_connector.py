import mysql.connector

from config.config_loader import ConfigLoader


class MySQLConnector:
    def __init__(self):
        config_file_path = '../config.json'
        config_loader = ConfigLoader(config_file_path)
        database = config_loader.get_config('database')
        self.mydb = mysql.connector.connect(
            host=database["host"],
            user=database["username"],
            password=database["password"],
            database="home_ai"
        )

    def get_pwm_configurations(self):
        cursor = self.mydb.cursor()
        cursor.execute("SELECT * FROM pwm_configuration")

        result = cursor.fetchall()
        return result
        # print(type(result))
        # for x in result:
        #     print(x)
        #     print(type(x))


if __name__ == '__main__':
    mysql = MySQLConnector()
    mysql.get_pwm_configurations()