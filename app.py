import os
import uuid
import time
import datetime
import base64
import io
import requests
import urllib.parse
from datetime import timezone
import threading
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm, CSRFProtect
# Use bootstrap-flask (Bootstrap5) instead of older flask-bootstrap
from flask_bootstrap import Bootstrap5
from werkzeug.utils import secure_filename
from wtforms import BooleanField, FileField, SubmitField, FloatField, HiddenField, StringField, TextAreaField, PasswordField
from wtforms.validators import Optional, Email, Length
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from itsdangerous import URLSafeTimedSerializer as Serializer

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std, preserve_color
from utils.payment_gateway import PLAN_PRICES, process_payment as run_dummy_payment


app = Flask(__name__)

# FIX: Force browser to not cache responses so CSS/HTML updates are immediately visible
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Secret key from environment variable — never hardcode in production.
# IMPORTANT: os.urandom(24) generates a NEW key on every restart, which logs out
# all active users (their sessions become invalid). Use a stable fallback string instead.
_default_secret = 'dev-insecure-key-change-in-production-abc123xyz'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', _default_secret)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
csrf = CSRFProtect(app)
login_manager = LoginManager()
login_manager.login_view = 'index'
login_manager.init_app(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(256))
    name = db.Column(db.String(1000))
    bio = db.Column(db.String(500), default='')
    avatar = db.Column(db.String(255), default='')
    plan = db.Column(db.String(50), default='free')
    email_verified = db.Column(db.Boolean, default=False)
    verify_token = db.Column(db.String(100), nullable=True)
    is_admin = db.Column(db.Boolean, default=False)
    transfers_today = db.Column(db.Integer, default=0)
    last_transfer_date = db.Column(db.Date, default=datetime.date.today)
    reset_token = db.Column(db.String(100), nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)
    payments = db.relationship('Payment', backref='user', lazy=True, cascade='all, delete-orphan')
    transfers = db.relationship('Transfer', backref='user', lazy=True, order_by='Transfer.created_at.desc()', cascade='all, delete-orphan')
    api_keys = db.relationship('ApiKey', backref='user', lazy=True, cascade='all, delete-orphan')
    likes = db.relationship('Like', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def get_reset_token(self):
        s = Serializer(app.config['SECRET_KEY'], salt='password-reset')
        return s.dumps({'user_id': self.id})

    @staticmethod
    def verify_reset_token(token, expires_sec=1800):
        s = Serializer(app.config['SECRET_KEY'], salt='password-reset')
        try:
            data = s.loads(token, max_age=expires_sec)
        except Exception:
            return None
        return db.session.get(User, data.get('user_id'))

    def get_verify_token(self):
        s = Serializer(app.config['SECRET_KEY'], salt='email-verify')
        return s.dumps({'user_id': self.id})

    @staticmethod
    def verify_email_token(token, expires_sec=86400):
        s = Serializer(app.config['SECRET_KEY'], salt='email-verify')
        try:
            data = s.loads(token, max_age=expires_sec)
        except Exception:
            return None
        return db.session.get(User, data.get('user_id'))


class Payment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.String(32), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    plan = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    card_last4 = db.Column(db.String(4))
    message = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(timezone.utc))


class Transfer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    content_image = db.Column(db.String(255), nullable=True)
    style_image = db.Column(db.String(255), nullable=True)
    result_image = db.Column(db.String(255), nullable=False)
    mask_image = db.Column(db.String(255), nullable=True)
    alpha = db.Column(db.Float, default=1.0)
    width = db.Column(db.Integer, default=0)
    height = db.Column(db.Integer, default=0)
    is_public = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(timezone.utc))
    likes = db.relationship('Like', backref='transfer', lazy='dynamic', cascade='all, delete-orphan')

    @property
    def likes_count(self):
        return self.likes.count()


class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    transfer_id = db.Column(db.Integer, db.ForeignKey('transfer.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(timezone.utc))
    __table_args__ = (db.UniqueConstraint('user_id', 'transfer_id', name='unique_user_transfer_like'),)


class Challenge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    theme = db.Column(db.String(200), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(timezone.utc))
    ends_at = db.Column(db.DateTime, nullable=False)
    submissions = db.relationship('ChallengeSubmission', backref='challenge', lazy='dynamic', cascade='all, delete-orphan')

    @property
    def submissions_count(self):
        return self.submissions.count()

    @property
    def is_expired(self):
        return datetime.datetime.now(timezone.utc).replace(tzinfo=None) > self.ends_at.replace(tzinfo=None)


class ChallengeSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    challenge_id = db.Column(db.Integer, db.ForeignKey('challenge.id'), nullable=False)
    transfer_id = db.Column(db.Integer, db.ForeignKey('transfer.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(timezone.utc))
    __table_args__ = (db.UniqueConstraint('challenge_id', 'user_id', name='unique_challenge_user_submission'),)
    transfer = db.relationship('Transfer', backref=db.backref('challenge_submissions', cascade='all, delete-orphan'))
    user = db.relationship('User', backref=db.backref('challenge_submissions', cascade='all, delete-orphan'))


class ApiKey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    key = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(100), default='Default')
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(timezone.utc))
    last_used = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)


PLAN_RESOLUTIONS = {
    'free': 512,
    'pro': 1024,
    'team': 2048,
}


def add_watermark(image_path, text='NeuralArt'):
    try:
        img = Image.open(image_path).convert('RGBA')
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        font_size = max(w, h) // 20
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        margin = font_size // 2
        positions = [
            (margin, margin),
            (w - tw - margin, margin),
            (margin, h - th - margin),
            (w - tw - margin, h - th - margin),
            (w // 2 - tw // 2, h // 2 - th // 2),
        ]
        for pos in positions:
            draw.text(pos, text, font=font, fill=(255, 255, 255, 80))
        watermarked = Image.alpha_composite(img, overlay).convert('RGB')
        watermarked.save(image_path, quality=95)
    except Exception:
        pass


def get_plan_resolution(plan):
    return PLAN_RESOLUTIONS.get(plan, 512)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

Bootstrap5(app)

with app.app_context():
    db.create_all()
    try:
        with db.engine.connect() as conn:
            conn.execute(db.text("ALTER TABLE transfer ADD COLUMN mask_image VARCHAR(255)"))
            conn.commit()
    except Exception:
        pass

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def _cleanup_old_uploads():
    """Remove unreferenced uploads older than 24 hours. Called on startup and every 30 min
    by background thread. Never removes files referenced in active database records."""
    now = time.time()
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
    except (OSError, FileNotFoundError):
        return  # Folder doesn't exist or became inaccessible
    
    try:
        active_files = set()
        for t in Transfer.query.all():
            if t.content_image: active_files.add(t.content_image)
            if t.style_image: active_files.add(t.style_image)
            if t.result_image: active_files.add(t.result_image)
            if getattr(t, 'mask_image', None): active_files.add(t.mask_image)
        for u in User.query.all():
            if u.avatar: active_files.add(u.avatar)
    except Exception as e:
        print(f"[WARN] Could not fetch active files for cleanup: {e}")
        return  # Abort cleanup if DB query fails to prevent accidental deletion
    
    for f in files:
        if f in active_files or f.startswith('preset_'):
            continue  # Protect database-linked images and curated presets forever!
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], f)
        try:
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > 86400:
                os.remove(fpath)
        except (OSError, FileNotFoundError):
            pass  # File deleted by concurrent request or another cleanup thread

# Background thread — runs cleanup every 30 min during app lifetime.
# Prevents disk fill-up from accumulated uploads without needing app restart.
def _start_cleanup_scheduler():
    def _loop():
        while True:
            time.sleep(30 * 60)  # sleep 30 minutes
            try:
                with app.app_context():
                    _cleanup_old_uploads()
            except Exception:
                pass  # Never crash the background thread
    t = threading.Thread(target=_loop, daemon=True, name='upload-cleanup')
    t.start()

STYLE_PRESETS = [
    {'id': 'mondrian', 'name': 'Mondrian Abstract', 'file': 'preset_mondrian.jpg', 'src': 'mondrian.jpg', 'desc': 'Geometric abstract art', 'pro': False, 'icon': 'fa-th-large'},
    {'id': 'matisse', 'name': 'Matisse Fauvism', 'file': 'preset_matisse.jpg', 'src': 'woman_with_hat_matisse.jpg', 'desc': 'Vibrant colorful portraiture', 'pro': False, 'icon': 'fa-palette'},
    {'id': 'brushstrokes', 'name': 'Dynamic Brush', 'file': 'preset_brushstrokes.jpg', 'src': 'brushstrokes.jpg', 'desc': 'Bold textured strokes', 'pro': False, 'icon': 'fa-paint-brush'},
    {'id': 'sketch', 'name': 'Pencil Sketch', 'file': 'preset_sketch.png', 'src': 'sketch.png', 'desc': 'Classic graphite sketch', 'pro': False, 'icon': 'fa-pencil-alt'},
    {'id': 'lamuse', 'name': 'Picasso La Muse', 'file': 'preset_lamuse.jpg', 'src': 'la_muse.jpg', 'desc': 'Cubist masterpiece', 'pro': False, 'icon': 'fa-shapes'},
    {'id': 'contrast', 'name': 'Contrast Forms', 'file': 'preset_contrast.jpg', 'src': 'contrast_of_forms.jpg', 'desc': 'Modernist geometric forms', 'pro': False, 'icon': 'fa-vector-square'},
    {'id': 'street', 'name': 'Parisian Street', 'file': 'preset_street.jpg', 'src': 'scene_de_rue.jpg', 'desc': 'Impressionist street view', 'pro': True, 'icon': 'fa-city'},
    {'id': 'flower', 'name': 'Flower of Life', 'file': 'preset_flower.jpg', 'src': 'flower_of_life.jpg', 'desc': 'Sacred geometry pattern', 'pro': True, 'icon': 'fa-sun'},
    {'id': 'reservoir', 'name': 'Poitiers Reservoir', 'file': 'preset_reservoir.jpg', 'src': 'the_resevoir_at_poitiers.jpg', 'desc': 'Post-impressionist landscape', 'pro': True, 'icon': 'fa-water'},
    {'id': 'antimono', 'name': 'Vibrant Mono', 'file': 'preset_antimono.jpg', 'src': 'antimonocromatismo.jpg', 'desc': 'Contemporary color field', 'pro': True, 'icon': 'fa-rainbow'},
]

def _init_style_presets():
    """Copy preset style images from style_data/ to static/uploads/ on startup."""
    import shutil
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        for p in STYLE_PRESETS:
            src_path = os.path.join('style_data', p['src'])
            dst_path = os.path.join(app.config['UPLOAD_FOLDER'], p['file'])
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)
    except Exception as e:
        print(f"[WARN] Failed to init style presets: {e}")

@app.context_processor
def inject_presets():
    return dict(style_presets=STYLE_PRESETS)

with app.app_context():
    _init_style_presets()   # Ensure presets exist in upload folder
    _cleanup_old_uploads()  # Immediate cleanup on startup
_start_cleanup_scheduler()   # Then every 30 min after

# Both files are required for style transfer (Optional validator for WTForms)
class UploadForm(FlaskForm):
    content = FileField('Content Image', validators=[Optional()])
    style = FileField('Style Image', validators=[Optional()])
    style2 = FileField('Second Style (Optional)', validators=[Optional()])
    content_path = HiddenField()
    style_path = HiddenField()
    style2_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    blend_ratio = FloatField('Style Blend Ratio', default=0.5)
    preserve_color = BooleanField('Preserve Original Colors', default=False)
    use_multiscale = BooleanField('Multi-scale Enhancement', default=False)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

try:
    encoder = VGGEncoder('vgg_normalised.pth').to(device)
    decoder = Decoder().to(device)
    # Aligned decoder filename to 'decoder.pth' instead of 'decoder_final.pth'
    decoder.load_state_dict(torch.load('decoder.pth', map_location=device, weights_only=True))
    encoder.eval()
    decoder.eval()
except Exception as e:
    print(f"[FATAL] Model loading failed: {e}")
    print("Run 'python setup.py' to download models.")
    raise

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def _transfer_at_scale(content_img, style_img, encoder, decoder, alpha, device, scale_size):
    """Run single-scale style transfer at given size."""
    content_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.ToTensor()
    ])
    style_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.ToTensor()
    ])
    content_tensor = content_transform(content_img).unsqueeze(0).to(device)
    style_tensor = style_transform(style_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # encoder(is_test=True) returns h4 tensor (B, 512, H/8, W/8)
        c_feats = encoder(content_tensor, is_test=True)
        s_feats = encoder(style_tensor, is_test=True)
        t_feats = adaptive_instance_normalization(c_feats, s_feats)
        t_feats = alpha * t_feats + (1 - alpha) * c_feats
        out_tensor = decoder(t_feats)

    out_img = out_tensor.cpu().squeeze(0).clamp(0, 1)
    return transforms.ToPILImage()(out_img)


def _get_adain_feats_at_scale(content_img, style_img, encoder, device, scale_size):
    """Extract AdaIN features at a specific scale for multi-scale blending.

    Resizes both images to scale_size and passes through the encoder.
    Returns the content and style feature maps (h4 layer only) for use
    in adaptive instance normalization and feature-level blending.

    Args:
        content_img: PIL Image (content).
        style_img: PIL Image (style).
        encoder: VGGEncoder model.
        device: torch device.
        scale_size: Target size to resize images to before encoding.

    Returns:
        (c_f, s_f): Tuple of h4 feature tensors, each (B, 512, H', W').
    """
    tfm = transforms.Compose([transforms.Resize(scale_size), transforms.ToTensor()])
    c_t = tfm(content_img).unsqueeze(0).to(device)
    s_t = tfm(style_img).unsqueeze(0).to(device)
    with torch.no_grad():
        # encoder(is_test=True) returns only h4 (relu4-1), not the full tuple
        c_f = encoder(c_t, is_test=True)
        s_f = encoder(s_t, is_test=True)
        return c_f, s_f


def _decode_feats(t_feats, decoder):
    """Decode features to PIL image."""
    with torch.no_grad():
        out_t = decoder(t_feats)
    out_img = out_t.cpu().squeeze(0).clamp(0, 1)
    return transforms.ToPILImage()(out_img)


def style_transfer(content_image, style_image, encoder, decoder, alpha, device, target_size=512, preserve_color=False, style_image2=None, blend_ratio=0.5, use_multiscale=False):
    # Clamp parameters to valid ranges to prevent corrupted output
    alpha = max(0.0, min(1.0, float(alpha)))
    blend_ratio = max(0.0, min(1.0, float(blend_ratio)))
    process_size = 512

    if preserve_color:
        # Resize to process_size FIRST so YCbCr channels match decoder output size
        resized_content = content_image.resize((process_size, process_size), Image.LANCZOS)
        content_ycbcr = resized_content.convert('YCbCr')
        content_y, content_cb, content_cr = content_ycbcr.split()
        transfer_input = content_y.convert('RGB')
    else:
        transfer_input = content_image

    if use_multiscale:
        # Multi-scale: feature-level AdaIN at 256px, decode at 512px
        c_feats_256, s_feats_256 = _get_adain_feats_at_scale(transfer_input, style_image, encoder, device, 256)
        c_feats_512, s_feats_512 = _get_adain_feats_at_scale(transfer_input, style_image, encoder, device, 512)

        t_feats_256 = adaptive_instance_normalization(c_feats_256, s_feats_256)
        t_feats_512 = adaptive_instance_normalization(c_feats_512, s_feats_512)

        t_feats_256 = alpha * t_feats_256 + (1 - alpha) * c_feats_256
        t_feats_512 = alpha * t_feats_512 + (1 - alpha) * c_feats_512

        # Upsample 256px features to match 512px spatial size before blending
        t_feats_256 = torch.nn.functional.interpolate(
            t_feats_256, size=t_feats_512.shape[2:], mode='bilinear', align_corners=False
        )
        # Blend features: low-res gives structure, high-res gives detail
        t_feats = 0.4 * t_feats_256 + 0.6 * t_feats_512

        if style_image2 is not None and blend_ratio > 0:
            _, s2_feats_512 = _get_adain_feats_at_scale(transfer_input, style_image2, encoder, device, 512)
            t2_feats = adaptive_instance_normalization(c_feats_512, s2_feats_512)
            t2_feats = alpha * t2_feats + (1 - alpha) * c_feats_512
            t_feats = (1 - blend_ratio) * t_feats + blend_ratio * t2_feats

        stylized_image = _decode_feats(t_feats, decoder)

    else:
        if style_image2 is not None and blend_ratio > 0:
            # Feature-level style interpolation (principled AdaIN blending)
            c_f, s1_f = _get_adain_feats_at_scale(transfer_input, style_image, encoder, device, process_size)
            _, s2_f = _get_adain_feats_at_scale(transfer_input, style_image2, encoder, device, process_size)

            t1 = adaptive_instance_normalization(c_f, s1_f)
            t2 = adaptive_instance_normalization(c_f, s2_f)
            t_feats = (1 - blend_ratio) * t1 + blend_ratio * t2
            t_feats = alpha * t_feats + (1 - alpha) * c_f
        else:
            c_f, s_f = _get_adain_feats_at_scale(transfer_input, style_image, encoder, device, process_size)
            t_feats = adaptive_instance_normalization(c_f, s_f)
            t_feats = alpha * t_feats + (1 - alpha) * c_f

        stylized_image = _decode_feats(t_feats, decoder)

    if preserve_color:
        stylized_y = stylized_image.convert('L')
        result = Image.merge('YCbCr', (stylized_y, content_cb, content_cr)).convert('RGB')
        stylized_image = result

    if target_size > process_size:
        w, h = stylized_image.size
        ratio = target_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        stylized_image = stylized_image.resize(new_size, Image.LANCZOS)

    return stylized_image


def save_image(image, path):
    image.save(path, quality=95)



@app.route('/signup', methods=['POST'])
def signup():
    email = (request.form.get('email') or '').strip()
    name = (request.form.get('fullname') or '').strip()
    password = request.form.get('password') or ''
    confirm_password = request.form.get('confirm_password') or ''

    if not email or not name or not password:
        flash('Please fill in all required fields.')
        return redirect(url_for('index') + '#auth')

    if password != confirm_password:
        flash('Passwords do not match.')
        return redirect(url_for('index') + '#auth')

    user = User.query.filter_by(email=email).first()
    if user:
        flash('Email address already exists.')
        return redirect(url_for('index') + '#auth')

    new_user = User(email=email, name=name, password=generate_password_hash(password, method='scrypt'), email_verified=False)
    db.session.add(new_user)
    db.session.commit()
    token = new_user.get_verify_token()
    verify_url = url_for('verify_email', token=token, _external=True)
    flash(f'Account created! Verify your email: {verify_url}', 'info')
    login_user(new_user)
    return redirect(url_for('index'))

@app.route('/login', methods=['POST'])
def login():
    email = (request.form.get('email') or '').strip()
    password = request.form.get('password') or ''

    if not email or not password:
        flash('Please enter both email and password.')
        return redirect(url_for('index') + '#auth')

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.')
        return redirect(url_for('index') + '#auth')

    login_user(user)
    return redirect(url_for('index'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/verify-email/<token>')
def verify_email(token):
    user = User.verify_email_token(token)
    if not user:
        flash('Invalid or expired verification link.', 'danger')
        return redirect(url_for('index'))
    if user.email_verified:
        flash('Email already verified.', 'info')
    else:
        user.email_verified = True
        db.session.commit()
        flash('Email verified successfully! You can now use all features.', 'success')
    return redirect(url_for('index'))

@app.route('/resend-verification')
@login_required
def resend_verification():
    if current_user.email_verified:
        flash('Email already verified.', 'info')
        return redirect(url_for('profile'))
    token = current_user.get_verify_token()
    verify_url = url_for('verify_email', token=token, _external=True)
    flash(verify_url, 'verification_link')
    return redirect(url_for('profile'))

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    transfer_id = None
    content_filename = None
    style_filename = None
    style2_filename = None
    mask_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            # Validate file type — reject invalid formats explicitly
            if allowed_file(form.content.data.filename):
                unique_id = uuid.uuid4().hex[:8]
                content_filename = f"{unique_id}_{secure_filename(form.content.data.filename)}"
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
            else:
                error = 'Content image must be PNG, JPG, or JPEG format.'
                return render_template('index.html', form=form, result_image=None,
                                       content_image=None, style_image=None, error=error)
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            # Validate file type — reject invalid formats explicitly
            if allowed_file(form.style.data.filename):
                unique_id = uuid.uuid4().hex[:8]
                style_filename = f"{unique_id}_{secure_filename(form.style.data.filename)}"
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
            else:
                error = 'Style image must be PNG, JPG, or JPEG format.'
                return render_template('index.html', form=form, result_image=None,
                                       content_image=content_filename, style_image=None, error=error)
        else:
            style_filename = form.style_path.data

        if form.style2.data and form.style2.data.filename:
            if allowed_file(form.style2.data.filename):
                unique_id = uuid.uuid4().hex[:8]
                style2_filename = f"{unique_id}_{secure_filename(form.style2.data.filename)}"
                form.style2.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style2_filename))
                form.style2_path.data = style2_filename
            else:
                error = 'Second style image must be PNG, JPG, or JPEG format.'
                return render_template('index.html', form=form, result_image=None,
                                       content_image=content_filename, style_image=style_filename, error=error)
        else:
            style2_filename = form.style2_path.data

        if style_filename and style_filename.startswith('preset_'):
            for p in STYLE_PRESETS:
                if p['file'] == style_filename and p.get('pro', False):
                    is_free = (current_user.is_authenticated and current_user.plan == 'free') or not current_user.is_authenticated
                    if is_free:
                        error = f"The '{p['name']}' style preset is a Pro/Team exclusive feature! Please subscribe to Pro to use this style."
                        return render_template('index.html', form=form, result_image=None, content_image=content_filename, style_image=style_filename, error=error)

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                # Validate image dimensions before processing (prevents OOM)
                MAX_DIM = 4096
                for img_obj, img_label in [(content_image, 'Content'), (style_image, 'Style')]:
                    w, h = img_obj.size
                    if w > MAX_DIM or h > MAX_DIM:
                        error = (f'{img_label} image is too large ({w}x{h}px). '
                                 f'Maximum allowed dimension is {MAX_DIM}px on either side.')
                        return render_template('index.html', form=form, result_image=None,
                                               content_image=content_filename,
                                               style_image=style_filename, error=error)

                # Safe alpha parsing — None check + clamp to [0.0, 1.0]
                try:
                    alpha = float(form.alpha.data) if form.alpha.data is not None else 1.0
                except (TypeError, ValueError):
                    alpha = 1.0
                alpha = max(0.0, min(1.0, alpha))  # Clamp: prevents out-of-range values
                # Check email verification
                if current_user.is_authenticated and not current_user.email_verified:
                    flash('Please verify your email before using style transfer.', 'warning')
                    return render_template('index.html', form=form, result_image=None, content_image=content_filename, style_image=style_filename, error=None)

                # Check subscription limits
                if not current_user.is_authenticated:
                    if 'transfers_today' not in session:
                        session['transfers_today'] = 0
                    if session['transfers_today'] >= 5:
                        error = 'Free limit reached (5 transfers/day). Please sign up and subscribe for unlimited transfers.'
                        return render_template('index.html', form=form, result_image=None, content_image=content_filename, style_image=style_filename, error=error)
                else:
                    if current_user.last_transfer_date != datetime.date.today():
                        current_user.last_transfer_date = datetime.date.today()
                        current_user.transfers_today = 0
                        db.session.commit()
                    if current_user.plan == 'free' and current_user.transfers_today >= 5:
                        error = 'Free limit reached (5 transfers/day). Please subscribe for unlimited transfers.'
                        return render_template('index.html', form=form, result_image=None, content_image=content_filename, style_image=style_filename, error=error)

                # Determine target resolution by plan
                if current_user.is_authenticated:
                    target_size = get_plan_resolution(current_user.plan)
                else:
                    target_size = 512

                do_preserve_color = form.preserve_color.data if form.preserve_color.data is not None else False
                do_multiscale = form.use_multiscale.data if form.use_multiscale.data is not None else False
                style2_image = None
                blend_ratio = 0.5
                if style2_filename:
                    style2_path = os.path.join(app.config['UPLOAD_FOLDER'], style2_filename)
                    style2_image = Image.open(style2_path).convert('RGB')
                    try:
                        blend_ratio = float(form.blend_ratio.data) if form.blend_ratio.data is not None else 0.5
                    except (TypeError, ValueError):
                        blend_ratio = 0.5
                # Clamp blend_ratio to valid range [0.0, 1.0]
                blend_ratio = max(0.0, min(1.0, blend_ratio))
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device, target_size, preserve_color=do_preserve_color, style_image2=style2_image, blend_ratio=blend_ratio, use_multiscale=do_multiscale)

                # Process Selective Style Mask if provided
                mask_data = request.form.get('mask_data')
                if mask_data and mask_data.startswith('data:image'):
                    try:
                        header, b64_str = mask_data.split(',', 1)
                        mask_bytes = base64.b64decode(b64_str)
                        mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
                        mask_img_resized = mask_img.resize(stylized_image.size, Image.Resampling.BILINEAR)
                        content_resized = content_image.resize(stylized_image.size, Image.Resampling.LANCZOS)
                        stylized_image = Image.composite(stylized_image, content_resized, mask_img_resized)
                        
                        unique_mask_id = uuid.uuid4().hex[:8]
                        mask_filename = f'mask_{unique_mask_id}.png'
                        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
                        mask_img_resized.save(mask_path, format='PNG')
                        print(f"[MASK] Selective style mask applied and saved: {mask_filename}")
                    except Exception as me:
                        print(f"[WARN] Failed to process selective mask: {me}")

                # UUID prefix prevents filename collision between users
                unique_id = uuid.uuid4().hex[:8]
                result_filename = f'stylized_{unique_id}_{content_filename}'
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)

                # Apply watermark for free tier users
                is_free = (current_user.is_authenticated and current_user.plan == 'free') or not current_user.is_authenticated
                if is_free:
                    add_watermark(result_path)

                # Increment transfer count ONLY after successful save + watermark
                if not current_user.is_authenticated:
                    session['transfers_today'] = session.get('transfers_today', 0) + 1
                    session.modified = True
                else:
                    current_user.transfers_today += 1
                    session.modified = True

                # Save transfer record
                if current_user.is_authenticated:
                    w, h = content_image.size
                    transfer = Transfer(
                        user_id=current_user.id,
                        content_image=content_filename,
                        style_image=style_filename,
                        result_image=result_filename,
                        mask_image=mask_filename,
                        alpha=alpha,
                        width=w,
                        height=h,
                        is_public=True,
                    )
                    db.session.add(transfer)
                    db.session.commit()
                    transfer_id = transfer.id
                
                result_image = result_filename
            except RuntimeError as e:
                # Expose a safe, user-friendly message. Do NOT leak internal tracebacks.
                msg = str(e)
                if 'out of memory' in msg.lower() or 'oom' in msg.lower():
                    error = 'Out of memory: image too large. Try a smaller image.'
                else:
                    error = 'Style transfer failed due to a processing error. Please try again.'
                print(f'[ERROR] Style transfer exception: {e}')  # Log internally
            except Exception as e:
                error = 'An unexpected error occurred. Please try again.'
                print(f'[ERROR] Unexpected exception: {e}')  # Log internally
        else:
            error = 'Please upload both content and style images.'
    elif request.method == 'POST':
        # This branch is hit when WTForms CSRF validation fails or form data is malformed
        error = 'Form submission failed (possible CSRF token mismatch). Please refresh the page and try again.'

    if content_filename:
        form.content_path.data = content_filename
    if style_filename:
        form.style_path.data = style_filename

    if current_user.is_authenticated:
        current_plan = current_user.plan
        res = get_plan_resolution(current_plan)
    else:
        current_plan = 'free'
        res = 512
    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error, current_plan=current_plan,
                           output_resolution=res, transfer_id=transfer_id, mask_image=mask_filename)


@app.route('/uploads/<filename>')
def send_image(filename):
    # Security check — prevent path traversal and serving arbitrary files
    # Only serve files whose extension matches allowed image types.
    safe_name = secure_filename(filename)
    if not safe_name or safe_name != filename:
        abort(400)  # Reject path traversal attempts (e.g. "../secret.txt")
    ext = safe_name.rsplit('.', 1)[-1].lower() if '.' in safe_name else ''
    if ext not in app.config['ALLOWED_EXTENSIONS']:
        abort(403)  # Reject non-image files even if they sneak into uploads/
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_name)


@app.route('/examples/<path:filename>')
def send_example(filename):
    safe_name = secure_filename(os.path.basename(filename))
    if not safe_name or safe_name != os.path.basename(filename):
        abort(400)
    ext = safe_name.rsplit('.', 1)[-1].lower() if '.' in safe_name else ''
    if ext not in app.config['ALLOWED_EXTENSIONS']:
        abort(403)
    try:
        # Use absolute path relative to app root to avoid CWD-dependent failures
        return send_from_directory(os.path.join(app.root_path, 'examples'), safe_name)
    except FileNotFoundError:
        return '', 404


@app.route('/style_preset/<path:filename>')
def send_style_preset(filename):
    """Serve a preset style image from style_data/ for the preset gallery."""
    safe_name = secure_filename(os.path.basename(filename))
    if not safe_name or safe_name != os.path.basename(filename):
        abort(400)
    ext = safe_name.rsplit('.', 1)[-1].lower() if '.' in safe_name else ''
    if ext not in app.config['ALLOWED_EXTENSIONS']:
        abort(403)
    try:
        return send_from_directory(os.path.join(app.root_path, 'style_data'), safe_name)
    except FileNotFoundError:
        return '', 404


@app.route('/checkout/<plan>')
def checkout(plan):
    if not current_user.is_authenticated:
        flash('Please log in or sign up to subscribe.')
        return redirect(url_for('index') + '#auth')
    if plan not in PLAN_PRICES:
        abort(404)
    return render_template(
        'checkout.html',
        plan=plan,
        amount=PLAN_PRICES[plan],
    )


@app.route('/process_payment/<plan>', methods=['POST'])
@login_required
def process_payment(plan):
    if plan not in PLAN_PRICES:
        abort(404)

    if current_user.plan == plan:
        flash(f'You are already on the {plan.capitalize()} plan.')
        return redirect(url_for('index') + '#pricing')

    name = request.form.get('card_name', '')
    card_number = request.form.get('card_number', '')
    expiry = request.form.get('expiry', '')
    cvv = request.form.get('cvv', '')

    result = run_dummy_payment(plan, name, card_number, expiry, cvv)
    amount = PLAN_PRICES[plan]

    payment = Payment(
        transaction_id=result.transaction_id,
        user_id=current_user.id,
        plan=plan,
        amount=amount,
        status=result.status,
        card_last4=result.card_last4 or None,
        message=result.message,
    )
    db.session.add(payment)

    try:
        if result.success:
            current_user.plan = plan
            db.session.commit()
            return redirect(url_for('payment_success', transaction_id=result.transaction_id))

        db.session.commit()
        flash(result.message)
        return redirect(url_for('payment_failed', transaction_id=result.transaction_id))
    except Exception:
        db.session.rollback()
        flash('A database error occurred. Please try again.')
        return redirect(url_for('index') + '#pricing')


@app.route('/payment/success/<transaction_id>')
@login_required
def payment_success(transaction_id):
    payment = Payment.query.filter_by(
        transaction_id=transaction_id,
        user_id=current_user.id,
    ).first_or_404()
    if payment.status != 'completed':
        abort(404)
    return render_template('payment_success.html', payment=payment)


@app.route('/payment/failed/<transaction_id>')
@login_required
def payment_failed(transaction_id):
    payment = Payment.query.filter_by(
        transaction_id=transaction_id,
        user_id=current_user.id,
    ).first_or_404()
    if payment.status == 'completed':
        return redirect(url_for('payment_success', transaction_id=transaction_id))
    return render_template('payment_failed.html', payment=payment)

@app.route('/dashboard')
@login_required
def dashboard():
    page = request.args.get('page', 1, type=int)
    per_page = 12
    pagination = Transfer.query.filter_by(user_id=current_user.id)\
        .order_by(Transfer.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    transfers = pagination.items
    total_transfers = Transfer.query.filter_by(user_id=current_user.id).count()
    return render_template('dashboard.html',
                           transfers=transfers,
                           pagination=pagination,
                           total_transfers=total_transfers)


@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)


@app.route('/profile/update', methods=['POST'])
@login_required
def profile_update():
    name = (request.form.get('name') or '').strip()
    bio = (request.form.get('bio') or '').strip()
    current_password = request.form.get('current_password') or ''
    new_password = request.form.get('new_password') or ''
    confirm_password = request.form.get('confirm_password') or ''

    if name:
        current_user.name = name
    if bio:
        current_user.bio = bio

    if new_password:
        if not current_password:
            flash('Please enter your current password to set a new one.')
            return redirect(url_for('profile'))
        if not check_password_hash(current_user.password, current_password):
            flash('Current password is incorrect.')
            return redirect(url_for('profile'))
        if new_password != confirm_password:
            flash('New passwords do not match.')
            return redirect(url_for('profile'))
        if len(new_password) < 6:
            flash('New password must be at least 6 characters.')
            return redirect(url_for('profile'))
        current_user.password = generate_password_hash(new_password, method='scrypt')

    db.session.commit()
    flash('Profile updated successfully!')
    return redirect(url_for('profile'))


@login_manager.unauthorized_handler
def unauthorized():
    flash('Please log in to access this page.')
    return redirect(url_for('index') + '#auth')


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip()
        user = User.query.filter_by(email=email).first()
        if user:
            token = user.get_reset_token()
            # In production, send actual email. Here we show the link directly.
            reset_url = url_for('reset_password', token=token, _external=True)
            flash(f'Password reset link: {reset_url}', 'info')
        else:
            flash('If that email is registered, a reset link has been sent.', 'info')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.verify_reset_token(token)
    if not user:
        flash('Invalid or expired reset token. Please request a new one.', 'danger')
        return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        password = request.form.get('password') or ''
        confirm = request.form.get('confirm_password') or ''
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('reset_password.html', token=token)
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('reset_password.html', token=token)
        user.password = generate_password_hash(password, method='scrypt')
        db.session.commit()
        flash('Password reset successfully! Please log in.', 'success')
        return redirect(url_for('index') + '#auth')
    return render_template('reset_password.html', token=token)


@app.route('/delete_transfer/<int:transfer_id>', methods=['POST'])
@login_required
def delete_transfer(transfer_id):
    transfer = Transfer.query.filter_by(id=transfer_id, user_id=current_user.id).first_or_404()
    # Remove TOCTOU race — call os.remove directly
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], transfer.result_image)
    try:
        os.remove(result_path)
    except OSError:
        pass  # File already deleted by concurrent request or cleanup thread
    db.session.delete(transfer)
    db.session.commit()
    flash('Transfer deleted.')
    return redirect(url_for('dashboard'))


# ── Admin decorator ──
def admin_required(f):
    from functools import wraps
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated


# ── Community Gallery ──
@app.route('/gallery')
def gallery():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    all_public = Transfer.query.filter_by(is_public=True)\
        .order_by(Transfer.created_at.desc()).all()
    upload_abs = os.path.abspath(app.config['UPLOAD_FOLDER'])
    transfers = [t for t in all_public if t.result_image and os.path.exists(os.path.join(upload_abs, t.result_image))]
    total = len(transfers)
    start = (page - 1) * per_page
    end = start + per_page
    transfers = transfers[start:end]
    class FakePagination:
        def __init__(self, items, total, page, per_page):
            self.items = items
            self.total = total
            self.page = page
            self.per_page = per_page
            self.pages = (total + per_page - 1) // per_page if total > 0 else 1
            self.has_prev = page > 1
            self.has_next = page < self.pages
            self.prev_num = page - 1
            self.next_num = page + 1
        def iter_pages(self, **kwargs):
            for p in range(1, self.pages + 1):
                yield p
    pagination = FakePagination(transfers, total, page, per_page)
    liked_ids = set()
    if current_user.is_authenticated:
        liked_ids = {l.transfer_id for l in Like.query.filter_by(user_id=current_user.id).all()}
    active_challenge = Challenge.query.filter_by(is_active=True).order_by(Challenge.created_at.desc()).first()
    return render_template('gallery.html', transfers=transfers, pagination=pagination, total=total, liked_ids=liked_ids, active_challenge=active_challenge)


@app.route('/artwork/<int:transfer_id>')
def artwork_detail(transfer_id):
    transfer = db.session.get(Transfer, transfer_id)
    if not transfer:
        abort(404)
    if not transfer.is_public and (not current_user.is_authenticated or (current_user.id != transfer.user_id and not getattr(current_user, 'is_admin', False))):
        abort(403)
    return render_template('artwork_detail.html', transfer=transfer)


@app.route('/toggle_public/<int:transfer_id>', methods=['POST'])
@login_required
def toggle_public(transfer_id):
    transfer = Transfer.query.filter_by(id=transfer_id, user_id=current_user.id).first_or_404()
    transfer.is_public = not transfer.is_public
    db.session.commit()
    flash(f'Transfer is now {"public" if transfer.is_public else "private"}.' if transfer.is_public else 'Transfer is now private.')
    return redirect(url_for('dashboard'))


@app.route('/like/<int:transfer_id>', methods=['POST'])
@login_required
def toggle_like(transfer_id):
    transfer = Transfer.query.get_or_404(transfer_id)
    if not transfer.is_public:
        abort(404)
    existing_like = Like.query.filter_by(user_id=current_user.id, transfer_id=transfer_id).first()
    if existing_like:
        db.session.delete(existing_like)
    else:
        db.session.add(Like(user_id=current_user.id, transfer_id=transfer_id))
    db.session.commit()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'likes_count': transfer.likes_count, 'liked': existing_like is None})
    return redirect(url_for('gallery'))


# ── Daily Challenges ──
@app.route('/challenges')
def challenges():
    active = Challenge.query.filter_by(is_active=True).order_by(Challenge.created_at.desc()).all()
    expired = Challenge.query.filter_by(is_active=False).order_by(Challenge.created_at.desc()).limit(5).all()
    return render_template('challenges.html', active=active, expired=expired)


@app.route('/challenge/<int:challenge_id>')
def challenge_detail(challenge_id):
    challenge = Challenge.query.get_or_404(challenge_id)
    page = request.args.get('page', 1, type=int)
    pagination = ChallengeSubmission.query.filter_by(challenge_id=challenge_id)\
        .order_by(ChallengeSubmission.created_at.desc())\
        .paginate(page=page, per_page=12, error_out=False)
    submissions = pagination.items
    user_submission = None
    if current_user.is_authenticated:
        user_submission = ChallengeSubmission.query.filter_by(
            challenge_id=challenge_id, user_id=current_user.id
        ).first()
    return render_template('challenge_detail.html', challenge=challenge,
                           submissions=submissions, pagination=pagination,
                           user_submission=user_submission)


@app.route('/challenge/<int:challenge_id>/submit', methods=['GET', 'POST'])
@login_required
def submit_to_challenge(challenge_id):
    challenge = Challenge.query.get_or_404(challenge_id)
    if challenge.is_expired:
        flash('This challenge has ended.', 'warning')
        return redirect(url_for('challenge_detail', challenge_id=challenge_id))
    existing = ChallengeSubmission.query.filter_by(
        challenge_id=challenge_id, user_id=current_user.id
    ).first()
    if existing:
        flash('You already submitted to this challenge.', 'warning')
        return redirect(url_for('challenge_detail', challenge_id=challenge_id))
    user_transfers = Transfer.query.filter_by(user_id=current_user.id, is_public=True)\
        .order_by(Transfer.created_at.desc()).all()
    if not user_transfers:
        flash('You need at least one public artwork to submit. Go create and share first!', 'warning')
        return redirect(url_for('challenge_detail', challenge_id=challenge_id))
    return render_template('challenge_submit.html', challenge=challenge, transfers=user_transfers)


@app.route('/challenge/<int:challenge_id>/confirm/<int:transfer_id>', methods=['POST'])
@login_required
def confirm_challenge_submission(challenge_id, transfer_id):
    challenge = Challenge.query.get_or_404(challenge_id)
    transfer = Transfer.query.filter_by(id=transfer_id, user_id=current_user.id).first_or_404()
    if challenge.is_expired:
        flash('This challenge has ended.', 'warning')
        return redirect(url_for('challenge_detail', challenge_id=challenge_id))
    existing = ChallengeSubmission.query.filter_by(
        challenge_id=challenge_id, user_id=current_user.id
    ).first()
    if existing:
        flash('You already submitted to this challenge.', 'warning')
        return redirect(url_for('challenge_detail', challenge_id=challenge_id))
    submission = ChallengeSubmission(
        challenge_id=challenge_id,
        transfer_id=transfer_id,
        user_id=current_user.id
    )
    db.session.add(submission)
    db.session.commit()
    flash('Your artwork has been submitted to the challenge!', 'success')
    return redirect(url_for('challenge_detail', challenge_id=challenge_id))


@app.route('/challenge/create', methods=['GET', 'POST'])
@login_required
def create_challenge():
    if not current_user.is_admin:
        abort(403)
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        description = request.form.get('description', '').strip()
        theme = request.form.get('theme', '').strip()
        days = request.form.get('days', 7, type=int)
        if not title or not description or not theme:
            flash('All fields are required.', 'danger')
            return redirect(url_for('create_challenge'))
        ends_at = datetime.datetime.now(timezone.utc) + datetime.timedelta(days=days)
        challenge = Challenge(
            title=title, description=description, theme=theme,
            ends_at=ends_at, is_active=True
        )
        db.session.add(challenge)
        db.session.commit()
        flash('Challenge created!', 'success')
        return redirect(url_for('challenges'))
    return render_template('create_challenge.html')


# ── Image Upscaler ──
@app.route('/upscale', methods=['GET', 'POST'])
def upscale():
    if request.method == 'POST':
        if 'image' not in request.files or not request.files['image'].filename:
            flash('Please upload an image.', 'danger')
            return redirect(url_for('upscale'))
        file = request.files['image']
        if not allowed_file(file.filename):
            flash('Only PNG, JPG, and JPEG files are allowed.', 'danger')
            return redirect(url_for('upscale'))
        scale = request.form.get('scale', '2', type=int)
        if scale not in (2, 4):
            scale = 2
        try:
            img = Image.open(file.stream).convert('RGB')
            w, h = img.size
            if w * scale > 4096 or h * scale > 4096:
                flash(f'Output would be too large ({w*scale}x{h*scale}). Max 4096px.', 'danger')
                return redirect(url_for('upscale'))
            new_w, new_h = w * scale, h * scale
            upscaled = img.resize((new_w, new_h), Image.LANCZOS)
            unique_id = uuid.uuid4().hex[:8]
            result_filename = f'upscaled_{unique_id}.png'
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            upscaled.save(result_path, 'PNG')
            return render_template('upscale_result.html',
                                   original_url=url_for('send_image', filename=secure_filename(file.filename)) if file.filename else '',
                                   result_url=url_for('send_image', filename=result_filename),
                                   result_filename=result_filename,
                                   original_size=f'{w}x{h}',
                                   new_size=f'{new_w}x{new_h}',
                                   scale=scale)
        except Exception as e:
            flash('Failed to process image. Please try again.', 'danger')
            return redirect(url_for('upscale'))
    return render_template('upscale.html')


# ── Text to Image ──
@app.route('/text-to-image', methods=['GET', 'POST'])
def text_to_image():
    if request.method == 'POST':
        prompt = request.form.get('prompt', '').strip()
        if not prompt:
            flash('Please enter a description.', 'danger')
            return redirect(url_for('text_to_image'))
        width = request.form.get('width', 512, type=int)
        height = request.form.get('height', 512, type=int)
        if width not in (256, 512, 768, 1024):
            width = 512
        if height not in (256, 512, 768, 1024):
            height = 512
        style = request.form.get('style', '')
        full_prompt = f"{prompt}, {style} style, high quality, detailed" if style else f"{prompt}, high quality, detailed"
        try:
            import urllib.parse
            resp = None
            for attempt in range(3):
                encoded = urllib.parse.quote(full_prompt)
                seed = int.from_bytes(os.urandom(4), 'big')
                # Try flux model first, fallback to standard on retries
                model_param = "&model=flux" if attempt == 0 else ""
                api_url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&seed={seed}&nologo=true{model_param}"
                print(f'[TXT2IMG] Attempt {attempt+1}: Requesting {full_prompt} ({width}x{height})')
                try:
                    r = requests.get(api_url, timeout=60, verify=False)
                    if r.status_code == 200 and len(r.content) > 1000 and 'image' in r.headers.get("Content-Type", ""):
                        resp = r
                        break
                    print(f'[TXT2IMG] Attempt {attempt+1} failed: status={r.status_code}, len={len(r.content)}')
                except Exception as ex:
                    print(f'[TXT2IMG] Attempt {attempt+1} error: {ex}')
                time.sleep(1.0)
            
            if not resp or resp.status_code != 200:
                print(f'[TXT2IMG] All attempts failed.')
                flash('Image generation temporarily busy. Please try again in a moment.', 'danger')
                return redirect(url_for('text_to_image'))
            
            unique_id = uuid.uuid4().hex[:8]
            result_filename = f'txt2img_{unique_id}.png'
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            with open(result_path, 'wb') as f:
                f.write(resp.content)
            print(f'[TXT2IMG] Saved: {result_filename} ({len(resp.content)} bytes)')
            transfer_id = None
            if current_user.is_authenticated:
                transfer = Transfer(
                    user_id=current_user.id,
                    content_image=result_filename,
                    style_image=result_filename,
                    result_image=result_filename,
                    alpha=1.0,
                    width=width,
                    height=height,
                    is_public=True,
                )
                db.session.add(transfer)
                db.session.commit()
                transfer_id = transfer.id
            return render_template('txt2img_result.html',
                                   result_url=url_for('send_image', filename=result_filename),
                                   result_filename=result_filename,
                                   prompt=full_prompt,
                                   width=width,
                                   height=height,
                                   transfer_id=transfer_id)
        except requests.Timeout:
            print('[TXT2IMG] Timeout')
            flash('Generation timed out. Please try again.', 'danger')
            return redirect(url_for('text_to_image'))
        except Exception as e:
            print(f'[TXT2IMG] Error: {type(e).__name__}: {e}')
            flash('Failed to generate image. Please try again.', 'danger')
            return redirect(url_for('text_to_image'))
    return render_template('txt2img.html')


# ── API Access ──
def generate_api_key():
    return 'nai_' + uuid.uuid4().hex + uuid.uuid4().hex[:16]


def get_api_user():
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        key = auth[7:]
        api_key = ApiKey.query.filter_by(key=key, is_active=True).first()
        if api_key:
            api_key.last_used = datetime.datetime.now(timezone.utc)
            db.session.commit()
            return api_key.user
    return None


@app.route('/api/v1/transfer', methods=['POST'])
@csrf.exempt
def api_transfer():
    api_user = get_api_user()
    if not api_user:
        return {'error': 'Invalid or missing API key. Use header: Authorization: Bearer <key>'}, 401
    if api_user.plan not in ('pro', 'team'):
        return {'error': 'API access requires Pro or Team plan.'}, 403
    if 'content' not in request.files or 'style' not in request.files:
        return {'error': 'Both "content" and "style" image files required.'}, 400
    content_file = request.files['content']
    style_file = request.files['style']
    if not content_file.filename or not style_file.filename:
        return {'error': 'Both files must have a filename.'}, 400
    if not allowed_file(content_file.filename) or not allowed_file(style_file.filename):
        return {'error': 'Files must be PNG, JPG, or JPEG.'}, 400

    alpha = request.form.get('alpha', '1.0')
    try:
        alpha = max(0.0, min(1.0, float(alpha)))
    except (TypeError, ValueError):
        alpha = 1.0

    target_size = get_plan_resolution(api_user.plan)

    try:
        content_img = Image.open(content_file).convert('RGB')
        style_img = Image.open(style_file).convert('RGB')

        # Rate limit check
        today = datetime.date.today()
        if api_user.last_transfer_date != today:
            api_user.last_transfer_date = today
            api_user.transfers_today = 0
        limit = 100 if api_user.plan == 'team' else 50
        if api_user.transfers_today >= limit:
            return {'error': f'Daily limit ({limit}) reached for your plan.'}, 429

        preserve_color_api = request.form.get('preserve_color', 'false').lower() in ('true', '1')
        stylized = style_transfer(content_img, style_img, encoder, decoder, alpha, device, target_size, preserve_color=preserve_color_api)

        api_user.transfers_today += 1
        api_user.last_transfer_date = today

        cid = uuid.uuid4().hex[:8]
        cfn = f'{cid}_{secure_filename(content_file.filename)}'
        sfn = f'{cid}_{secure_filename(style_file.filename)}'
        rfn = f'api_{cid}_{secure_filename(content_file.filename)}'
        content_file.seek(0)
        content_file.save(os.path.join(app.config['UPLOAD_FOLDER'], cfn))
        style_file.seek(0)
        style_file.save(os.path.join(app.config['UPLOAD_FOLDER'], sfn))
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], rfn)
        save_image(stylized, result_path)

        if api_user.plan == 'free':
            add_watermark(result_path)

        transfer = Transfer(
            user_id=api_user.id,
            content_image=cfn,
            style_image=sfn,
            result_image=rfn,
            alpha=alpha,
            width=content_img.width,
            height=content_img.height,
        )
        db.session.add(transfer)
        db.session.commit()

        return {
            'success': True,
            'result_url': url_for('send_image', filename=rfn, _external=True),
            'transfer_id': transfer.id,
            'resolution': f'{stylized.width}x{stylized.height}',
        }
    except Exception as e:
        return {'error': f'Processing failed: {str(e)}'}, 500


@app.route('/api/v1/enhance_prompt', methods=['GET', 'POST'])
@csrf.exempt
def api_enhance_prompt():
    prompt = request.values.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'Prompt cannot be empty'}), 400
    
    lower_p = prompt.lower()
    style_modifiers = []
    
    if any(k in lower_p for k in ['cyber', 'neon', 'future', 'robot', 'city', 'sci-fi', 'tokyo']):
        style_modifiers = ['cyberpunk aesthetic', 'neon glow', 'wet pavement reflections', 'cinematic lighting', 'futuristic metropolis', 'octane render']
    elif any(k in lower_p for k in ['dragon', 'knight', 'castle', 'magic', 'wizard', 'elf', 'sword', 'myth']):
        style_modifiers = ['epic fantasy digital painting', 'ethereal volumetric lighting', 'dramatic composition', 'highly detailed armor and textures', 'unreal engine 5 render']
    elif any(k in lower_p for k in ['girl', 'boy', 'man', 'woman', 'face', 'portrait', 'eyes', 'model']):
        style_modifiers = ['stunning cinematic portrait', 'soft Rembrandt lighting', 'photorealistic skin texture', 'shallow depth of field', '85mm lens', '8k resolution']
    elif any(k in lower_p for k in ['river', 'mountain', 'lake', 'sunset', 'forest', 'tree', 'sea', 'ocean', 'sky']):
        style_modifiers = ['breathtaking landscape photography', 'golden hour sunlight', 'atmospheric mist', 'National Geographic style', 'vibrant color palette', 'hyper-detailed']
    else:
        style_modifiers = ['masterpiece', 'highly detailed', 'dramatic lighting', 'cinematic composition', 'award-winning concept art', '8k resolution', 'unreal engine 5 render']
    
    if len(prompt.split()) > 15:
        enhanced = prompt + ", " + ", ".join(style_modifiers[:2])
    else:
        enhanced = f"{prompt}, " + ", ".join(style_modifiers)
        
    return jsonify({'success': True, 'original_prompt': prompt, 'enhanced_prompt': enhanced})


@app.route('/api/v1/keys', methods=['GET', 'POST'])
@login_required
def api_keys():
    if current_user.plan not in ('pro', 'team'):
        flash('API access requires Pro or Team plan.', 'warning')
        return redirect(url_for('index') + '#pricing')
    if request.method == 'POST':
        name = (request.form.get('name') or '').strip() or f'Key {len(current_user.api_keys) + 1}'
        key_str = generate_api_key()
        api_key = ApiKey(user_id=current_user.id, key=key_str, name=name)
        db.session.add(api_key)
        db.session.commit()
        flash(f'API key created: {key_str}', 'info')
        return redirect(url_for('api_keys'))
    return render_template('api_docs.html', keys=current_user.api_keys)


@app.route('/api/v1/keys/<int:key_id>/delete', methods=['POST'])
@login_required
def delete_api_key(key_id):
    api_key = ApiKey.query.filter_by(id=key_id, user_id=current_user.id).first_or_404()
    db.session.delete(api_key)
    db.session.commit()
    flash('API key deleted.')
    return redirect(url_for('api_keys'))


# ── Admin Dashboard ──
@app.route('/admin')
@admin_required
def admin_dashboard():
    stats = {
        'total_users': User.query.count(),
        'total_transfers': Transfer.query.count(),
        'total_payments': Payment.query.count(),
        'total_public': Transfer.query.filter_by(is_public=True).count(),
        'users_today': User.query.filter(User.last_transfer_date == datetime.date.today()).count(),
        'transfers_today': Transfer.query.filter(
            db.func.date(Transfer.created_at) == datetime.date.today()
        ).count(),
        'revenue': db.session.query(db.func.sum(Payment.amount)).filter_by(status='completed').scalar() or 0,
    }
    recent_users = User.query.order_by(User.id.desc()).limit(10).all()
    recent_transfers = Transfer.query.order_by(Transfer.created_at.desc()).limit(10).all()
    return render_template('admin_dashboard.html', stats=stats,
                           recent_users=recent_users, recent_transfers=recent_transfers)


@app.route('/admin/users')
@admin_required
def admin_users():
    page = request.args.get('page', 1, type=int)
    search = (request.args.get('search') or '').strip()
    query = User.query
    if search:
        query = query.filter(
            db.or_(User.name.ilike(f'%{search}%'), User.email.ilike(f'%{search}%'))
        )
    pagination = query.order_by(User.id.desc()).paginate(page=page, per_page=20, error_out=False)
    return render_template('admin_users.html', users=pagination.items, pagination=pagination, search=search)


@app.route('/admin/transfers')
@admin_required
def admin_transfers():
    page = request.args.get('page', 1, type=int)
    pagination = Transfer.query.order_by(Transfer.created_at.desc())\
        .paginate(page=page, per_page=20, error_out=False)
    return render_template('admin_transfers.html', transfers=pagination.items, pagination=pagination)


@app.route('/admin/payments')
@admin_required
def admin_payments():
    page = request.args.get('page', 1, type=int)
    pagination = Payment.query.order_by(Payment.created_at.desc())\
        .paginate(page=page, per_page=20, error_out=False)
    return render_template('admin_payments.html', payments=pagination.items, pagination=pagination)


@app.route('/admin/user/<int:user_id>/toggle-admin', methods=['POST'])
@admin_required
def admin_toggle_admin(user_id):
    if user_id == current_user.id:
        flash('You cannot remove your own admin status.', 'danger')
        return redirect(url_for('admin_users'))
    user = db.session.get(User, user_id)
    if not user:
        abort(404)
    user.is_admin = not user.is_admin
    db.session.commit()
    flash(f'Admin status toggled for {user.name}.')
    return redirect(url_for('admin_users'))


@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    if user_id == current_user.id:
        flash('You cannot delete yourself.', 'danger')
        return redirect(url_for('admin_users'))
    user = db.session.get(User, user_id)
    if not user:
        abort(404)
    Transfer.query.filter_by(user_id=user.id).delete()
    Payment.query.filter_by(user_id=user.id).delete()
    ApiKey.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.name} deleted.')
    return redirect(url_for('admin_users'))


@app.route('/admin/transfer/<int:transfer_id>/delete', methods=['POST'])
@admin_required
def admin_delete_transfer(transfer_id):
    transfer = db.session.get(Transfer, transfer_id)
    if not transfer:
        abort(404)
    # Remove TOCTOU race — call os.remove directly
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], transfer.result_image)
    try:
        os.remove(result_path)
    except OSError:
        pass  # File already deleted by concurrent request or cleanup thread
    db.session.delete(transfer)
    db.session.commit()
    flash('Transfer deleted.')
    return redirect(url_for('admin_transfers'))


@app.errorhandler(403)
def forbidden(e):
    if current_user.is_authenticated:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    return redirect(url_for('index') + '#auth')


@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    is_debug = os.environ.get('FLASK_DEBUG') == '1'
    app.run(host='0.0.0.0', port=5000, debug=True)
