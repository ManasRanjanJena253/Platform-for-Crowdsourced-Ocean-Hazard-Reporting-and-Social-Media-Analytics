import asyncio

from motor.motor_asyncio import AsyncIOMotorClient

client = AsyncIOMotorClient(host = "localhost", port = 27017)

db_name = "Crowd_Sourced_Ocean_Hazard_Reporting"


# Collections to be made :
# user : Stores all the data related to the user
# reports : Stores all the information related to the reports sent by the users
# hotspot : Stores all the data related to the hotspot usage in a particular area

user_validation = {
    "$jsonSchema":
        {"bsonType": "object",
         "required": ["user_name", "user_id", "role", "hashed_pwd"],
         "properties": {
             "user_name": {
                 "bsonType": "string",
                 "description" : "The name of the user"
             },
             "user_id": {
                 "bsonType": "string",
                 "description": "unique id provided to the user"
             },
             "role": {
                 "bsonType": "string",
                 "description": "The role of the user i.e. an official or a normal citizen reporting."
             },
             "hashed_pwd": {
                 "bsonType": "string",
                 "description": "The hashed password of the user for security."
             }
         }}
}

reports_validation = {
    "$jsonSchema":
        {"bsonType": "object",
         "required": ["user_id", "report_id", "location", "timestamp", "report_url"],
         "properties":{
             "user_id": {
                 "bsonType": "string",
                 "description": "Uniques user id or the name of the media being submitted"
             },
             "report_id": {
                 "bsonType": "string",
                 "description": "Unique id given to a particular report"
             },
             "location": {
                 "bsonType": "object",
                 "required": ["latitude", "longitude"],
                 "properties": {
                     "latitude": {"bsonType": "double"},
                     "longitude": {"bsonType": "double"}
                 }
             },
             "timestamp": {
                 "bsonType": "date",
                 "description": "The time when the report was submitted"
             },
             "ai_tags": {
                 "bsonType": "object",
                 "required": ["classification", "urgency"],
                 "properties": {
                     "classification": {
                         "bsonType": "string",
                         "description": "The type of calamity or disaster"
                     },
                     "urgency": {
                         "bsonType": "string",
                         "description": "The issue resolving urgency"
                     }
                 }
             },
             "source": {
                 "bsonType": "string",
                 "description": "The source used for acquiring the data i.e. Citizen or Social Media"
             }
         }}
}

hotspot_validation = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["hotspot_id", "location_center", "report_count"],
        "properties": {
            "hotspot_id": {
                "bsonType": "string",
                "description": "The unique hotspot id of the citizens"
            },
            "location_center": {
                "bsonType": "string",
                "description": "The location center under which the location of hotspot falls into"
            },
            "report_count": {"bsonType": "int",
            "description": "The no. of reports recieved from that particular location_center to which the hotspot belongs to ."
                             },

            "urgency_level": {
                "bsonType": "double",
                "description": "The urgency level of the hazard based on DBSCAN or HDBSCAN."
            }
        }
    }
}

async def create_collections():
    # await client.drop_database(db_name)
    db = client[db_name]
    collections = ["user", "hotspot", "reports"]
    validators = [user_validation, hotspot_validation, reports_validation]
    built_collections = await db.list_collections()
    built_collections_list = await built_collections.to_list()

    for idx, coll in enumerate(collections):
        if coll in built_collections_list:
            print(f"The collection {coll} already exists.")
        else:
            try:
                await db.create_collection(
                    name = coll,
                    validator = validators[idx],
                    validationAction = "error"
                )
                print(f"Collection {coll} created successfully")
            except Exception as e:
                print("Error : ", e)

if __name__ == "__main__":
    asyncio.run(create_collections())
