from flask import Flask
import pymongo
import json

app = Flask(__name__)

client = pymongo.MongoClient(host='127.0.0.1', port=27017)
db = client.get_database(name='zhihu')
user_col = db.get_collection('user')
proxy_col = db.get_collection('proxy')
relationship_col = db.get_collection('relationship')
history_list = []


@app.route('/zhihu_user_count')
def index():
    history_list.insert(0, [user_col.count(), relationship_col.count(), proxy_col.count()])
    return json.dumps(
        history_list,
        indent=2
    )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
