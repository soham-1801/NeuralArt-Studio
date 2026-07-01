import os
import uuid
import time
import datetime
from datetime import timezone
import threading
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm, CSRFProtect
# FIX #22: Use bootstrap-flask (Bootstrap5) instead of older flask-bootstrap
from flask_bootstrap import Bootstrap5
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import Optional
from PIL import Image
from torchvision import transforms

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std
from utils.payment_gateway import PLAN_PRICES, process_payment as run_dummy_payment


app = Flask(__name__)

# FIX: Force browser to not cache responses so CSS/HTML updates are immediately visible
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# FIX #2: Secret key from environment variable — never hardcode in production.
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
    password = db.Column(db.String(256))  # scrypt hashes are ~130+ chars; 256 gives safe headroom
    name = db.Column(db.String(1000))
    plan = db.Column(db.String(50), default='free')
    transfers_today = db.Column(db.Integer, default=0)
    last_transfer_date = db.Column(db.Date, default=datetime.date.today)
    payments = db.relationship('Payment', backref='user', lazy=True)


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

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

Bootstrap5(app)

with app.app_context():
    db.create_all()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def _cleanup_old_uploads():
    """FIX #6: Cleanup old uploads — called explicitly & by background thread.
    Avoids side-effects when app is imported by testing tools or WSGI workers."""
    now = time.time()
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], f)
        if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > 3600:
            try:
                os.remove(fpath)
            except OSError:
                pass  # Another worker may have deleted it

# FIX #12: Background thread — runs cleanup every 30 min during app lifetime.
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

with app.app_context():
    _cleanup_old_uploads()  # Immediate cleanup on startup
_start_cleanup_scheduler()   # Then every 30 min after

# FIX #9: Added DataRequired-style validators — both files are now explicitly required
class UploadForm(FlaskForm):
    content = FileField('Content Image', validators=[Optional()])
    style = FileField('Style Image', validators=[Optional()])
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

try:
    encoder = VGGEncoder('vgg_normalised.pth').to(device)
    decoder = Decoder().to(device)
    # FIX #23: Aligned decoder filename to 'decoder.pth' instead of 'decoder_final.pth'
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

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)



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

    new_user = User(email=email, name=name, password=generate_password_hash(password, method='scrypt'))
    db.session.add(new_user)
    db.session.commit()
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

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            # FIX #3: Explicit error if file type is invalid (was silently skipped before)
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
            # FIX #3: Explicit error if file type is invalid (was silently skipped before)
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

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                # FIX #13: Validate image dimensions before processing.
                # Very large images cause OOM spikes even after Resize(512).
                MAX_DIM = 4096
                for img_obj, img_label in [(content_image, 'Content'), (style_image, 'Style')]:
                    w, h = img_obj.size
                    if w > MAX_DIM or h > MAX_DIM:
                        error = (f'{img_label} image is too large ({w}x{h}px). '
                                 f'Maximum allowed dimension is {MAX_DIM}px on either side.')
                        return render_template('index.html', form=form, result_image=None,
                                               content_image=content_filename,
                                               style_image=style_filename, error=error)

                # FIX #7: Safe alpha parsing — None check + clamp to [0.0, 1.0]
                try:
                    alpha = float(form.alpha.data) if form.alpha.data is not None else 1.0
                except (TypeError, ValueError):
                    alpha = 1.0
                alpha = max(0.0, min(1.0, alpha))  # Clamp: prevents out-of-range values
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

                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                # Increment transfer count
                if not current_user.is_authenticated:
                    session['transfers_today'] += 1
                    session.modified = True
                else:
                    current_user.transfers_today += 1
                    db.session.commit()

                # FIX #1: Use UUID prefix to prevent filename collision between users
                unique_id = uuid.uuid4().hex[:8]
                result_filename = f'stylized_{unique_id}_{content_filename}'
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)
                
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

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    # FIX #14: Security check — prevent path traversal and serving arbitrary files.
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

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    # FIX #24: Do not force use_debugger=True. Use environment variable.
    is_debug = os.environ.get('FLASK_DEBUG') == '1'
    run_simple('0.0.0.0', 5000, app, use_reloader=is_debug, use_debugger=is_debug)





