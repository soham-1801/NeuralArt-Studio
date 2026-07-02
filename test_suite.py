import unittest
import app as a
import re
import io
import os
import base64
from PIL import Image

class NeuralArtTestSuite(unittest.TestCase):
    def setUp(self):
        self.app = a.app
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for programmatic testing
        self.client = self.app.test_client()
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()

    def _create_dummy_image_bytes(self, color='red', size=(100, 100)):
        img = Image.new('RGB', size, color=color)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        return buf

    def test_01_public_routes(self):
        routes = [
            '/',
            '/gallery',
            '/challenges',
            '/upscale',
            '/text-to-image',
            '/forgot-password',
        ]
        for route in routes:
            resp = self.client.get(route)
            self.assertEqual(resp.status_code, 200, f"Failed GET {route} with status {resp.status_code}")
            print(f"[OK] GET {route} -> {resp.status_code}")

    def test_02_404_and_errors(self):
        resp = self.client.get('/nonexistent_route_12345')
        self.assertEqual(resp.status_code, 404)
        self.assertIn('Page Not Found', resp.get_data(as_text=True))
        print("[OK] GET /nonexistent_route_12345 -> 404")

    def test_03_auth_and_user_actions(self):
        email = 'test_user_qa@neuralart.ai'
        password = 'testpassword123'
        
        u = a.User.query.filter_by(email=email).first()
        if u:
            a.db.session.delete(u)
            a.db.session.commit()

        # Signup
        resp = self.client.post('/signup', data={
            'email': email,
            'fullname': 'QA Tester',
            'password': password,
            'confirm_password': password
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200, "Signup failed")
        print("[OK] POST /signup -> 200")

        u = a.User.query.filter_by(email=email).first()
        self.assertIsNotNone(u, "User was not saved to database")
        print(f"[OK] User created in DB with ID: {u.id}")

        # Protected routes when free
        for route in ['/dashboard', '/profile', '/checkout/pro', '/checkout/team']:
            resp = self.client.get(route)
            self.assertEqual(resp.status_code, 200, f"Failed accessing protected route {route}")
            print(f"[OK] GET {route} (authenticated) -> {resp.status_code}")

        # Profile update
        resp = self.client.post('/profile/update', data={
            'name': 'QA Tester Updated',
            'email': email
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200, "Profile update failed")
        print("[OK] POST /profile/update -> 200")

        # Test dummy payment processing
        resp = self.client.post('/process_payment/pro', data={
            'card_name': 'QA Tester',
            'card_number': '4242 4242 4242 4242',
            'expiry': '12/30',
            'cvv': '123'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200, "Payment processing failed")
        print("[OK] POST /process_payment/pro -> 200")

        a.db.session.refresh(u)
        self.assertEqual(u.plan, 'pro')
        print("[OK] User plan upgraded to PRO")

        # Test API keys route when upgraded to pro -> 200
        resp = self.client.get('/api/v1/keys')
        self.assertEqual(resp.status_code, 200, "Failed GET /api/v1/keys after PRO upgrade")
        print("[OK] GET /api/v1/keys (PRO plan) -> 200")

        # Create API key
        resp = self.client.post('/api/v1/keys', data={'name': 'QA Key'}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200, "Failed creating API key")
        print("[OK] POST /api/v1/keys -> 200")

        a.db.session.refresh(u)
        api_key_obj = u.api_keys[0] if u.api_keys else None
        self.assertIsNotNone(api_key_obj)
        print(f"[OK] Created API Key: {api_key_obj.key}")

        # Test API transfer using the new API key
        resp = self.client.post('/api/v1/transfer', headers={
            'Authorization': f'Bearer {api_key_obj.key}'
        }, data={
            'content': (self._create_dummy_image_bytes('blue'), 'content.jpg'),
            'style': (self._create_dummy_image_bytes('yellow'), 'style.jpg'),
            'alpha': '0.8'
        })
        self.assertEqual(resp.status_code, 200, f"API style transfer failed: {resp.get_data(as_text=True)}")
        print("[OK] POST /api/v1/transfer with Bearer token -> 200")

        # Test UI Style Transfer (POST to '/')
        resp = self.client.post('/', data={
            'content': (self._create_dummy_image_bytes('green'), 'content_ui.jpg'),
            'style': (self._create_dummy_image_bytes('purple'), 'style_ui.jpg'),
            'alpha': '1.0'
        })
        self.assertEqual(resp.status_code, 200, f"UI style transfer failed with status {resp.status_code}")
        self.assertIn('stylized_', resp.get_data(as_text=True))
        print("[OK] POST / (UI style transfer) -> 200")

        # Check if transfer record was saved
        t = a.Transfer.query.filter_by(user_id=u.id).first()
        self.assertIsNotNone(t, "Transfer record not saved in DB")
        print(f"[OK] Transfer record created in DB with ID: {t.id}")

        # Test toggle public (returns 302 redirect)
        resp = self.client.post(f'/toggle_public/{t.id}')
        self.assertEqual(resp.status_code, 302, "Toggle public failed")
        print(f"[OK] POST /toggle_public/{t.id} -> 302")

        # Test like (returns JSON when X-Requested-With is set)
        resp = self.client.post(f'/like/{t.id}', headers={'X-Requested-With': 'XMLHttpRequest'})
        self.assertEqual(resp.status_code, 200, "Toggle like failed")
        print(f"[OK] POST /like/{t.id} -> 200")

        # Test challenge submission
        ch = a.Challenge.query.first()
        if ch:
            resp = self.client.post(f'/challenge/{ch.id}/submit', data={'transfer_id': t.id}, follow_redirects=True)
            self.assertEqual(resp.status_code, 200, "Challenge submission failed")
            print(f"[OK] POST /challenge/{ch.id}/submit -> 200")

        # Test delete transfer
        resp = self.client.post(f'/delete_transfer/{t.id}', follow_redirects=True)
        self.assertEqual(resp.status_code, 200, "Delete transfer failed")
        print(f"[OK] POST /delete_transfer/{t.id} -> 200")

        # Logout
        resp = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(resp.status_code, 200, "Logout failed")
        print("[OK] GET /logout -> 200")

    def test_04_admin_routes(self):
        resp = self.client.get('/admin')
        self.assertEqual(resp.status_code, 302, "Expected redirect for unauthenticated admin access")
        print("[OK] GET /admin (unauthenticated) -> 302")

        admin_email = 'admin@neuralart.ai'
        u = a.User.query.filter_by(email=admin_email).first()
        if not u:
            print("[WARN] admin@neuralart.ai not found in db, skipping admin tests")
            return
        
        u.password = a.generate_password_hash('admin123')
        a.db.session.commit()

        self.client.post('/login', data={'email': admin_email, 'password': 'admin123'})
        for route in ['/admin', '/admin/users', '/admin/transfers', '/admin/payments']:
            resp = self.client.get(route)
            self.assertEqual(resp.status_code, 200, f"Admin route {route} failed with {resp.status_code}")
            print(f"[OK] GET {route} (admin) -> {resp.status_code}")

    def test_05_challenge_and_transfer_endpoints(self):
        admin_email = 'admin@neuralart.ai'
        self.client.post('/login', data={'email': admin_email, 'password': 'admin123'})
        
        resp = self.client.get('/challenges')
        self.assertEqual(resp.status_code, 200)
        print("[OK] GET /challenges -> 200")

        ch = a.Challenge.query.first()
        if ch:
            resp = self.client.get(f'/challenge/{ch.id}')
            self.assertEqual(resp.status_code, 200, f"Failed GET /challenge/{ch.id}")
            print(f"[OK] GET /challenge/{ch.id} -> {resp.status_code}")

    def test_06_phase1_realworld_features(self):
        # 1. Verify Prompt Enhancer API
        resp_empty = self.client.post('/api/v1/enhance_prompt', data={'prompt': ''})
        self.assertEqual(resp_empty.status_code, 400, "Empty prompt should return 400")
        
        resp_dragon = self.client.post('/api/v1/enhance_prompt', data={'prompt': 'a dragon flying over a medieval castle'})
        self.assertEqual(resp_dragon.status_code, 200, "Valid prompt should return 200")
        data_dragon = resp_dragon.get_json()
        self.assertTrue(data_dragon['success'])
        self.assertIn('epic fantasy', data_dragon['enhanced_prompt'])
        print(f"[OK] POST /api/v1/enhance_prompt -> 200 ({data_dragon['enhanced_prompt'][:40]}...)")

        # 2. Verify Presets Initialization and Route
        self.assertGreater(len(a.STYLE_PRESETS), 0, "STYLE_PRESETS list should not be empty")
        for p in a.STYLE_PRESETS[:3]:
            resp_preset = self.client.get(f"/style_preset/{p['src']}")
            self.assertEqual(resp_preset.status_code, 200, f"Preset {p['src']} should be served with 200 OK")
        print("[OK] Style presets verified and serving correctly")

    def test_07_phase2_social_sharing(self):
        transfer = a.Transfer.query.filter_by(is_public=True).first()
        if transfer:
            resp = self.client.get(f'/artwork/{transfer.id}')
            self.assertEqual(resp.status_code, 200, f"Failed GET /artwork/{transfer.id}")
            html = resp.get_data(as_text=True)
            self.assertIn('og:image', html, "OpenGraph og:image meta tag not found in artwork detail HTML")
            self.assertIn('twitter:card', html, "Twitter card meta tag not found in artwork detail HTML")
            print(f"[OK] GET /artwork/{transfer.id} -> 200 with OpenGraph & Twitter Card meta tags verified")
        
        resp_404 = self.client.get('/artwork/999999')
        self.assertEqual(resp_404.status_code, 404, "Non-existent artwork should return 404")
        print("[OK] GET /artwork/999999 -> 404 correctly handled")

    def test_08_phase3_selective_masking(self):
        self.client.post('/login', data={'email': 'admin@neuralart.ai', 'password': 'admin123'}, follow_redirects=True)
        mask_img = Image.new('L', (100, 100), color=255)
        buf = io.BytesIO()
        mask_img.save(buf, format='PNG')
        b64_str = base64.b64encode(buf.getvalue()).decode('ascii')
        mask_data_url = f"data:image/png;base64,{b64_str}"

        data = {
            'content': (self._create_dummy_image_bytes('blue'), 'test_content_mask.jpg'),
            'style': (self._create_dummy_image_bytes('purple'), 'test_style_mask.jpg'),
            'alpha': '1.0',
            'mask_data': mask_data_url
        }
        resp = self.client.post('/', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(resp.status_code, 200, f"POST / with mask_data should succeed with 200 OK: {resp.get_data(as_text=True)[:200]}")
        
        latest_transfer = a.Transfer.query.order_by(a.Transfer.id.desc()).first()
        self.assertIsNotNone(latest_transfer, "Transfer should be recorded in database")
        self.assertIsNotNone(latest_transfer.mask_image, f"mask_image filename should be stored in DB. Latest transfer ID={latest_transfer.id}, mask_image={latest_transfer.mask_image}, content={latest_transfer.content_image}")
        
        mask_path = os.path.join(self.app.config['UPLOAD_FOLDER'], latest_transfer.mask_image)
        self.assertTrue(os.path.exists(mask_path), f"Mask file {mask_path} should exist on disk")
        print(f"[OK] POST / with Selective Style Mask -> 200 (Mask DB & Disk verification passed: {latest_transfer.mask_image})")

if __name__ == '__main__':
    unittest.main()
