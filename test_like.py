import app as a
import re

with a.app.app_context():
    with a.app.test_client() as client:
        # Get CSRF token
        resp = client.get('/gallery')
        match = re.search(r'name="csrf_token" value="([^"]+)"', resp.get_data(as_text=True))
        csrf = match.group(1) if match else ''
        print('CSRF token found:', bool(csrf))

        # Login
        resp_login = client.post('/login', data={'email': 'sohammangroliya778@gmail.com', 'password': 'soham123', 'csrf_token': csrf}, follow_redirects=True)
        print('Login status:', resp_login.status_code)

        # Like with CSRF
        resp = client.post('/like/1', headers={'X-Requested-With': 'XMLHttpRequest', 'X-CSRFToken': csrf})
        print('Status:', resp.status_code)
        print('Data:', resp.get_data(as_text=True))

        # Like again (unlike)
        resp = client.post('/like/1', headers={'X-Requested-With': 'XMLHttpRequest', 'X-CSRFToken': csrf})
        print('Status:', resp.status_code)
        print('Data:', resp.get_data(as_text=True))
