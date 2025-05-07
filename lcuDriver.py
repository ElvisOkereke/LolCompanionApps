from lcu_driver import Connector

connector = Connector()  # Use your certificate

@connector.ready
async def connect(connection):
    summoner = await connection.request("get", "/lol-summoner/v1/current-summoner")
    if summoner.status == 200:
        print("Success! Current Summoner:", (await summoner.json())["gameName"])

connector.start()