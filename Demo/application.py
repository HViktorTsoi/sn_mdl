from flask import Flask
import redis
import json

app = Flask(__name__)


@app.route('/start')
def start():
    history_list.insert(0, [user_col.count(), relationship_col.count(), proxy_col.count()])
    return json.dumps(
        history_list,
        indent=2
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
