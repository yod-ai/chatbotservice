Create a Python virtual environment.

python -m venv venv 

source venv/bin/activate

Run: pip install -r requirements.txt

Send a Request through postman after running app.py (Flask server):

1. Open Postman
	•	Select POST as the request type.
	•	Enter the URL: http://127.0.0.1:3010/chat
2. Go to the “Body” Tab
	•	Select raw and choose JSON as the format.
	•	Enter the following JSON payload: 

{
    "user_id": "user_123",
    "session_id": "session_456",
    "query": "What kind of food is best for my dog?"
}

3. Go to “Headers”
    •	Add the following header:

Content-Type: application/json

4. Click send

OR

For testing use the main loop directly in the v4.py file. 