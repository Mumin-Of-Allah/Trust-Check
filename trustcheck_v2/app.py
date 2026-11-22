# app.py — TrustCheck v2 (All 8 Features Implemented)
# =====================================================
# NEW in v2:
#   [F1] Real ML models (RandomForest) — loaded from model.joblib + scaler.joblib
#   [F2] Real EXE confidence — VirusTotal hash lookup replaces random.uniform()
#   [F3] SHAP explainability — top-5 feature attributions on phishing scans
#   [F4] Dynamic Risk Scoring (DRS) — formalized weighted formula
#   [F5] Isolation Forest behavioral anomaly detection on login sessions
#   [F6] Honeyword authentication — detects credential theft
#   [F7] Flask-SocketIO real-time alerts — WebSocket push replaces flash()
#   [F8] AES-256 Fernet encryption — scan results encrypted before DB write

import os, re, io, json, random, string, hashlib, tempfile, signal
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import numpy as np
import requests as http_requests
import pandas as pd
import psutil
import joblib

from cryptography.fernet import Fernet
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session, send_file
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.secret_key = os.environ.get("FLASK_SECRET", "supersecretkey_change_me")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app_data.db'
app.config['MAIL_SERVER']         = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT']           = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS']        = True
app.config['MAIL_USERNAME']       = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD']       = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', app.config['MAIL_USERNAME'])

# [F7] SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                    logger=False, engineio_logger=False)

ALLOWED_EXTENSIONS = {'log', 'txt', 'exe'}

VT_API_KEY      = os.environ.get("VT_API_KEY", "")
GSB_API_KEY     = os.environ.get("GSB_API_KEY", "")
URLSCAN_API_KEY = os.environ.get("URLSCAN_API_KEY", "")

db   = SQLAlchemy(app)
mail = Mail(app)

# [F8] Fernet AES-256
_raw_key = os.environ.get("FERNET_KEY", "")
try:
    fernet = Fernet(_raw_key.encode()) if _raw_key else Fernet(Fernet.generate_key())
except Exception:
    fernet = Fernet(Fernet.generate_key())

def encrypt_text(text):
    try: return fernet.encrypt(text.encode()).decode()
    except: return text

def decrypt_text(token):
    try: return fernet.decrypt(token.encode()).decode()
    except: return token


# ── [F1] ML Feature Extraction ───────────────────────────────────────────
SUSPICIOUS_KEYWORDS = [
    'paypal','ebay','netflix','amazon','apple','microsoft','google','facebook',
    'instagram','whatsapp','twitter','linkedin','dropbox','onedrive','icloud',
    'password','passwd','login','signin','sign-in','logon','account','secure',
    'security','verify','verification','confirm','update','upgrade','suspend',
    'suspended','unlock','reactivate','validate','authenticate','credential',
    'urgent','alert','warning','limited','expire','expiry','immediately',
    'prize','winner','congratulations','free-gift','claim','reward','bonus',
    'click-here','click-now','act-now','limited-time','offer','discount',
    'verify-account','confirm-identity','billing','invoice','refund','payment',
    'bank','credit','debit','card','wallet','crypto','bitcoin','transfer',
    'irs','tax','government','official','support','helpdesk',
]

def extract_url_features(url):
    try:
        p = urlparse(url)
        domain = p.netloc.lower().replace('www.', '')
        return [
            len(url),
            1 if re.match(r'^\d+\.\d+\.\d+\.\d+', domain) else 0,
            1 if '@' in url else 0,
            domain.count('.'),
            1 if '-' in domain else 0,
            0 if p.scheme == 'https' else 1,
            len(p.path),
            url.count('/'),
            url.count('?'),
            len(p.query),
            sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url.lower()),
            len(domain),
            1 if url.count('.') >= 4 else 0,
            sum(c.isdigit() for c in domain),
            1 if len(url) > 75 else 0,
        ]
    except:
        return [0] * 15

FEATURE_NAMES = [
    'url_length','has_ip','has_at_symbol','num_dots','has_hyphen',
    'no_https','path_length','num_slashes','num_query_params',
    'query_length','keyword_count','domain_length','multi_subdomain',
    'digits_in_domain','long_url_flag'
]

clf = scaler = None
_model_loaded = False
if os.path.exists("model.joblib") and os.path.exists("scaler.joblib"):
    try:
        clf    = joblib.load("model.joblib")
        scaler = joblib.load("scaler.joblib")
        _model_loaded = True
        print("[ML] Model loaded.")
    except Exception as e:
        print(f"[ML] Load failed: {e}")

# [F3] SHAP
shap_explainer = None
if _model_loaded:
    try:
        import shap as _shap
        shap_explainer = _shap.TreeExplainer(clf)
        print("[SHAP] Ready.")
    except Exception as e:
        print(f"[SHAP] Init failed: {e}")

lookup_df = None
if os.path.exists("data_lookup.pkl"):
    try: lookup_df = pd.read_pickle("data_lookup.pkl")
    except: pass

def featurize_number(num_str):
    s = str(num_str)
    return [int(s[-4:]), int(s[:3]), sum(int(c) for c in s if c.isdigit()), len(s)]

def allowed_file(f): return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_mobile(number):
    n = re.sub(r'[\s\-().]+', '', str(number))
    if n.startswith('+91'): n = n[3:]
    elif n.startswith('91') and len(n) == 12: n = n[2:]
    elif n.startswith('0') and len(n) == 11:  n = n[1:]
    return n.strip()

def generate_otp(): return ''.join(random.choices(string.digits, k=6))


# ═══ DATABASE MODELS ═══════════════════════════════════════════════════════

class User(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    full_name      = db.Column(db.String(120), nullable=False)
    state          = db.Column(db.String(60),  nullable=True)
    gender         = db.Column(db.String(20),  nullable=True)
    aadhaar_number = db.Column(db.String(12),  unique=True, nullable=False)
    mobile_number  = db.Column(db.String(15),  nullable=False)
    email          = db.Column(db.String(120), unique=True, nullable=True)
    username       = db.Column(db.String(80),  unique=True, nullable=True)
    password_hash  = db.Column(db.String(255), nullable=True)
    is_verified    = db.Column(db.Boolean, default=False)
    otp            = db.Column(db.String(64),  nullable=True)
    otp_expires_at = db.Column(db.DateTime,    nullable=True)
    created_at     = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    honeywords     = db.Column(db.Text, nullable=True)  # [F6] JSON list of hashed fake passwords

class PhishingHistory(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    url           = db.Column(db.Text,       nullable=False)   # [F8] encrypted
    status        = db.Column(db.String(50), nullable=False)
    confidence    = db.Column(db.Float,      nullable=True)    # [F4] DRS score
    ml_score      = db.Column(db.Float,      nullable=True)    # [F1]
    drs_score_val = db.Column(db.Float,      nullable=True)    # [F4]
    shap_top5     = db.Column(db.Text,       nullable=True)    # [F3] JSON
    created_at    = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class ExeHistory(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    exe_name   = db.Column(db.String(255), nullable=False)
    hash       = db.Column(db.String(64),  nullable=False)
    verdict    = db.Column(db.String(50),  nullable=False)
    confidence = db.Column(db.String(20),  nullable=True)
    vt_engines = db.Column(db.Integer,     nullable=True)  # [F2]
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class ProcessHistory(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    pid        = db.Column(db.Integer,     nullable=False)
    name       = db.Column(db.String(120), nullable=False)
    cpu        = db.Column(db.Float,       nullable=True)
    memory     = db.Column(db.Float,       nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class CallerHistory(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    number     = db.Column(db.String(15),  nullable=False)
    name       = db.Column(db.String(120), nullable=True)
    verified   = db.Column(db.Boolean,     default=False)
    gender     = db.Column(db.String(10),  nullable=True)
    checked_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class DeviceStatus(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    device_id  = db.Column(db.String(64),  nullable=False)
    status     = db.Column(db.String(10),  nullable=False, default='offline')
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class AadhaarRecord(db.Model):
    __tablename__ = 'aadhaar_records'
    id             = db.Column(db.Integer, primary_key=True)
    aadhaar_number = db.Column(db.String(12), unique=True, nullable=False)
    name           = db.Column(db.String(120), nullable=False)
    mobile         = db.Column(db.String(10),  nullable=False)
    state          = db.Column(db.String(60),  nullable=False)
    gender         = db.Column(db.String(10),  nullable=False)

class SessionEvent(db.Model):  # [F5]
    __tablename__ = 'session_events'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    event_type  = db.Column(db.String(50), nullable=False)
    ip_address  = db.Column(db.String(45), nullable=True)
    hour_of_day = db.Column(db.Integer,    nullable=True)
    day_of_week = db.Column(db.Integer,    nullable=True)
    timestamp   = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class HoneywordAttempt(db.Model):  # [F6]
    __tablename__ = 'honeyword_attempts'
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    ip_address   = db.Column(db.String(45), nullable=True)
    attempted_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

with app.app_context():
    db.create_all()
    if AadhaarRecord.query.count() == 0:
        seeds = [
            AadhaarRecord(aadhaar_number='123456789012', name='Ravi Kumar',   mobile='9876543210', state='Karnataka',   gender='Male'),
            AadhaarRecord(aadhaar_number='987654321098', name='Priya Singh',  mobile='8765432109', state='Tamil Nadu',  gender='Female'),
            AadhaarRecord(aadhaar_number='456789123456', name='Arjun Sharma', mobile='7654321098', state='Maharashtra', gender='Male'),
        ]
        db.session.bulk_save_objects(seeds)
        db.session.commit()


# ═══ [F5] BEHAVIORAL ANOMALY DETECTION ════════════════════════════════════

def log_session_event(user_id, event_type):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    db.session.add(SessionEvent(user_id=user_id, event_type=event_type,
        ip_address=request.remote_addr, hour_of_day=now.hour,
        day_of_week=now.weekday(), timestamp=now))
    db.session.commit()

def detect_behavioral_anomaly(user_id):
    events = SessionEvent.query.filter_by(user_id=user_id).all()
    if len(events) < 20:
        return {'anomaly': False, 'score': 0.0, 'reason': 'Insufficient history'}
    try:
        from sklearn.ensemble import IsolationForest as IF
        X = np.array([[e.hour_of_day or 0, e.day_of_week or 0,
                       hash(e.ip_address or '') % 1000] for e in events])
        iso = IF(contamination=0.1, random_state=42, n_estimators=100)
        iso.fit(X)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cur = np.array([[now.hour, now.weekday(), hash(request.remote_addr or '') % 1000]])
        score = float(iso.decision_function(cur)[0])
        is_anomaly = iso.predict(cur)[0] == -1
        return {'anomaly': bool(is_anomaly), 'score': round(score, 4),
                'reason': 'Unusual login time or IP' if is_anomaly else 'Normal behavior'}
    except Exception as e:
        return {'anomaly': False, 'score': 0.0, 'reason': str(e)}


# ═══ [F4] DYNAMIC RISK SCORING ════════════════════════════════════════════

def drs_score(pa, ts, ac, ca, w1=0.50, w2=0.40, w3=0.05, w4=0.05):
    """
    Dynamic Risk Score: w1*Pa + w2*Ts + w3*Ac + w4*Ca
    Smart fallback: if all external APIs fail (return 0),
    rely on the ML model score directly so phishing is still caught.
    """
    api_total = pa + ts + ac
    if api_total == 0:
        # APIs unavailable — use ML score as primary signal
        return min(100, int(ca))
    else:
        # APIs returned data — use weighted formula with stronger ML weight
        return min(100, int(0.40*pa + 0.35*ts + 0.05*ac + 0.20*ca))


# ═══ PHISHING SCAN APIs ════════════════════════════════════════════════════

def _virustotal_check(url):
    try:
        h = {"x-apikey": VT_API_KEY}
        r = http_requests.post("https://www.virustotal.com/api/v3/urls",
                               headers=h, data={"url": url}, timeout=10)
        if r.status_code != 200: return 0, "VT submission failed"
        aid = r.json()["data"]["id"]
        rep = http_requests.get(f"https://www.virustotal.com/api/v3/analyses/{aid}",
                                headers=h, timeout=10).json()
        mal = rep["data"]["attributes"]["stats"].get("malicious", 0)
        return min(mal * 10, 100), f"VT malicious engines: {mal}"
    except Exception as e: return 0, f"VT error: {e}"

def _google_safe_browsing(url):
    try:
        ep = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GSB_API_KEY}"
        payload = {"client":{"clientId":"trustcheck","clientVersion":"2.0"},
                   "threatInfo":{"threatTypes":["MALWARE","SOCIAL_ENGINEERING","UNWANTED_SOFTWARE"],
                                 "platformTypes":["ANY_PLATFORM"],"threatEntryTypes":["URL"],
                                 "threatEntries":[{"url":url}]}}
        r = http_requests.post(ep, json=payload, timeout=10)
        if r.status_code != 200: return 0, "GSB unavailable"
        if "matches" in r.json(): return 100, "Google Safe Browsing flagged this URL"
        return 0, "GSB: clean"
    except Exception as e: return 0, f"GSB error: {e}"

def _urlscan_check(url):
    try:
        h = {"API-Key": URLSCAN_API_KEY, "Content-Type": "application/json"}
        r = http_requests.post("https://urlscan.io/api/v1/scan/", headers=h,
                               json={"url": url, "visibility": "public"}, timeout=10)
        return (5, "URLScan submitted") if r.status_code == 200 else (0, "URLScan unavailable")
    except Exception as e: return 0, f"URLScan error: {e}"


# ═══ URL SCANNER PIPELINE ══════════════════════════════════════════════════

# ─── Known phishing / malware / IP-logger domains blacklist ────────────────
# Hardcoded high-confidence entries (phishing + IP loggers)
_HARDCODED_BLACKLIST = {
    # Phishing / malware
    'phishing-site.com', 'malware-download.net', 'fake-paypal.com',
    'secure-login-verify.com', 'account-suspended-alert.com',
    'click-here-prize.net', 'free-gift-claim.org',
    'verify-account-now.net', 'confirm-identity-urgently.com',
    'login-secure-update.com', 'account-verify-alert.net',
    'paypal-security-check.com', 'netflix-billing-update.net',
    'amazon-order-alert.com', 'apple-id-locked.net',
    'microsoft-support-alert.com', 'google-account-suspended.net',
    'bankofamerica-verify.com', 'chase-security-alert.net',
    'wellsfargo-account-update.com', 'citibank-verify-now.net',
    'irs-tax-refund-claim.com', 'covid19-relief-fund.net',
    'free-iphone-winner.com', 'prize-claim-now.net',
    'bitcoin-giveaway-elon.com', 'crypto-doubler.net',
    'urgent-delivery-notice.com', 'fedex-redelivery-fee.net',
    'dhl-parcel-held.com', 'usps-package-alert.net',
    'docusign-secure-view.com', 'adobe-sign-verify.net',
    'dropbox-shared-file.com', 'sharepoint-access-request.net',
    # IP loggers / tracking grabbers
    'grabify.link', 'iplogger.org', 'iplogger.com', 'iplogger.ru',
    '2no.co', 'yip.su', 'blasze.tk', 'ps3cfw.com',
    'discord-gifts.ml', 'bit.do', 'p.li', 'zackzack.de',
    'ipgrabber.ru', 'lovelocator.net', 'spoofbox.com',
    'whatismyipaddress.com', 'iptracker.org', 'canarytokens.com',
    'trackurl.it', 'redirect.li', 'cleanuri.com',
    'shortid.link', 'www.urltrends.com',
}

def _load_phishing_blacklist():
    """
    Build a blacklist of 10,000+ domains by combining:
      1. The hardcoded high-confidence set above.
      2. Verified-online phishing domains from verified_online.csv
         (first column = URL/domain, up to 15,000 entries).
    Returns a frozenset for O(1) lookup.
    """
    bl = set(_HARDCODED_BLACKLIST)
    csv_path = os.path.join(os.path.dirname(__file__), 'verified_online.csv')
    if os.path.exists(csv_path):
        try:
            import csv as _csv
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = _csv.reader(f)
                count = 0
                for row in reader:
                    if not row:
                        continue
                    raw = row[0].strip().lower()
                    # Strip scheme if present
                    raw = re.sub(r'^https?://', '', raw)
                    raw = re.sub(r'^www\.', '', raw)
                    domain = raw.split('/')[0].split(':')[0]
                    if domain and '.' in domain:
                        bl.add(domain)
                        count += 1
                        if count >= 15000:
                            break
        except Exception:
            pass
    return frozenset(bl)

# Build once at startup
DOMAIN_BLACKLIST = _load_phishing_blacklist()

# Known short URL services
SHORT_URL_DOMAINS = {
    'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly',
    'short.io', 'is.gd', 'buff.ly', 'adf.ly', 'tiny.cc',
    'rb.gy', 'cutt.ly', 'shorturl.at', 'snip.ly',
}

def extract_domain(url):
    """Step 2: Extract domain from URL."""
    try:
        p = urlparse(url if '://' in url else 'http://' + url)
        domain = p.netloc.lower().replace('www.', '')
        return domain.split(':')[0]
    except:
        return ''

def check_blacklist(domain):
    """Step 3: Check domain against known phishing blacklist."""
    return domain in DOMAIN_BLACKLIST

def expand_short_url(url):
    """Step 4: Expand shortened URLs by following redirects (HEAD request)."""
    try:
        domain = extract_domain(url)
        if domain not in SHORT_URL_DOMAINS:
            return url, False  # Not a short URL
        resp = http_requests.head(url, allow_redirects=True, timeout=6)
        final = resp.url
        return (final, True) if final != url else (url, False)
    except:
        return url, False

def get_domain_age(domain):
    """Step 5: Estimate domain age using WHOIS (or fallback heuristic)."""
    try:
        import whois as _whois
        w = _whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation:
            age_days = (datetime.now() - creation).days
            return age_days, creation.strftime('%Y-%m-%d')
    except:
        pass
    # Heuristic fallback: check for numeric/hyphenated new-looking domains
    suspicious_new = bool(re.search(r'\d{4,}', domain) or domain.count('-') >= 2)
    return (30 if suspicious_new else 365), 'Unknown'

def check_api_external(url):
    """Step 6: Run VirusTotal + GSB + URLScan checks."""
    vt_score,  vt_msg  = _virustotal_check(url)
    gsb_score, gsb_msg = _google_safe_browsing(url)
    us_score,  us_msg  = _urlscan_check(url)
    return {
        'vt_score': vt_score, 'vt_msg': vt_msg,
        'gsb_score': gsb_score, 'gsb_msg': gsb_msg,
        'us_score': us_score, 'us_msg': us_msg,
    }

def run_ml_prediction(url):
    """Step 7: Run Random Forest model prediction."""
    if not (_model_loaded and clf and scaler):
        return 0.0, [], False
    try:
        feats = extract_url_features(url)
        X_scaled = scaler.transform([feats])
        ml_score = round(float(clf.predict_proba(X_scaled)[0][1]) * 100, 1)
        # SHAP explanations
        shap_explanations = []
        if shap_explainer is not None:
            try:
                sv = shap_explainer.shap_values(X_scaled)
                vals = sv[1][0] if isinstance(sv, list) else sv[0]
                top5 = sorted(zip(FEATURE_NAMES, vals, feats),
                              key=lambda x: abs(x[1]), reverse=True)[:5]
                shap_explanations = [
                    {'feature': f, 'impact': round(float(v), 4), 'value': round(float(r), 2)}
                    for f, v, r in top5
                ]
            except:
                pass
        return ml_score, shap_explanations, True
    except Exception as e:
        return 0.0, [], False

def combine_scores(vt, gsb, us, ml, heuristic, domain_age_days, blacklisted):
    """Step 9: Dynamic Risk Scoring (DRS) — combine all signals into final score."""
    if blacklisted:
        return 95

    # Domain freshness penalty
    age_penalty = 0
    if domain_age_days is not None:
        if domain_age_days < 90:
            age_penalty = 25
        elif domain_age_days < 365:
            age_penalty = 10

    api_total = vt + gsb + us

    if api_total == 0 and ml == 0:
        # APIs down AND model not loaded — rely entirely on heuristics + age
        base = min(100, int(heuristic * 1.2))
    elif api_total == 0:
        # APIs down — rely on ML + heuristics
        base = int(ml * 0.65 + heuristic * 0.35)
    else:
        # Full scoring with all signals
        base = int(0.38 * vt + 0.34 * gsb + 0.05 * us + 0.18 * ml + 0.05 * heuristic)

    return min(100, base + age_penalty)

def final_classify(score):
    """Step 9: Final classification from score."""
    if score >= 70:
        return 'Phishing likely', '🚨', 'phishing'
    elif score >= 40:
        return 'Suspicious', '⚠️', 'suspicious'
    return 'Safe', '✅', 'safe'

def run_pipeline(url):
    """
    Full URL scanner pipeline:
    URL → Extract Domain → Blacklist Check → Expand Short URL →
    Domain Age → Feature Extraction → API Check → ML Prediction →
    Combine Scores → Final Classification
    """
    pipeline_steps = []

    # Step 1: Normalize URL
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    pipeline_steps.append({'step': 'Normalize URL', 'detail': url, 'status': 'ok'})

    # Step 2: Extract Domain
    domain = extract_domain(url)
    pipeline_steps.append({'step': 'Extract Domain', 'detail': domain or 'Failed', 'status': 'ok' if domain else 'warn'})

    # Step 3: Blacklist Check
    blacklisted = check_blacklist(domain)
    pipeline_steps.append({'step': 'Blacklist Check', 'detail': '🔴 Blacklisted!' if blacklisted else '✅ Not Blacklisted',
                            'status': 'danger' if blacklisted else 'ok'})
    if blacklisted:
        score = 95
        verdict, verdict_icon, verdict_cls = final_classify(score)
        # Pad remaining steps as skipped
        for skipped in ['Expand Short URL', 'Check Domain Age', 'External API Checks',
                        'AI/ML Model (Random Forest — 15 features)', 'SHAP Explanation',
                        'Dynamic Risk Scoring (DRS)', 'Final Classification']:
            pipeline_steps.append({'step': skipped, 'detail': 'Skipped — domain blacklisted', 'status': 'warn'})
        # Override last step
        pipeline_steps[-1] = {'step': 'Final Classification', 'detail': f'{verdict_icon} {verdict} (Blacklisted)', 'status': 'danger'}
        return {
            'url': url, 'domain': domain, 'score': score, 'verdict': verdict,
            'verdict_icon': verdict_icon, 'verdict_cls': verdict_cls,
            'blacklisted': True, 'expanded': False, 'expanded_url': url,
            'domain_age_days': None, 'domain_age_display': 'Unknown',
            'vt_score': 0, 'gsb_score': 0, 'us_score': 0,
            'ml_score': 0.0, 'model_loaded': _model_loaded,
            'signals': ['Domain is in known phishing/IP-logger blacklist'],
            'shap_explanations': [], 'pipeline_steps': pipeline_steps,
            'heuristic_score': 0,
        }

    # Step 4: Expand Short URL
    expanded_url, was_expanded = expand_short_url(url)
    if was_expanded:
        url = expanded_url
        domain = extract_domain(url)
        pipeline_steps.append({'step': 'Expand Short URL', 'detail': f'Revealed → {url[:60]}', 'status': 'warn'})
    else:
        pipeline_steps.append({'step': 'Expand Short URL', 'detail': 'Not a short URL — skipped', 'status': 'ok'})

    # Step 5: Domain Age
    domain_age_days, age_display = get_domain_age(domain)
    age_label = f'{domain_age_days}d old ({age_display})' if age_display != 'Unknown' else 'Unknown'
    age_status = 'warn' if domain_age_days < 90 else 'ok'
    age_note = ' ⚠️ New domain (<90 days)' if domain_age_days < 90 else ''
    pipeline_steps.append({'step': 'Check Domain Age', 'detail': age_label + age_note, 'status': age_status})

    # Heuristics (internal, no separate step)
    signals = []
    heuristic_score = 0
    p = urlparse(url)
    domain_part = domain.split(':')[0]

    if re.match(r'^\d+\.\d+\.\d+\.\d+$', domain_part):
        signals.append('IP address used as domain'); heuristic_score += 40
    if '@' in url:
        signals.append('@ symbol in URL'); heuristic_score += 30
    if p.scheme != 'https':
        signals.append('No HTTPS'); heuristic_score += 20
    if domain.count('-') >= 2:
        signals.append('Multiple hyphens in domain'); heuristic_score += 15
    if len(url) > 75:
        signals.append(f'Very long URL ({len(url)} chars)'); heuristic_score += 10
    if domain.count('.') >= 3:
        signals.append(f'Many subdomains ({domain.count(".")} dots)'); heuristic_score += 15
    if sum(c.isdigit() for c in domain_part) >= 4:
        signals.append('Many digits in domain'); heuristic_score += 10
    kw_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url.lower())
    if kw_hits >= 1:
        signals.append(f'{kw_hits} suspicious keyword(s) in URL')
        heuristic_score += min(kw_hits * 10, 40)
    if domain_age_days < 90:
        signals.append(f'Very new domain ({domain_age_days} days old)')

    # Feature extraction (internal, feeds ML)
    feats = extract_url_features(url)

    # Step 6: External API Checks
    api = check_api_external(url)
    api_all_failed = api['vt_score'] == 0 and api['gsb_score'] == 0 and api['us_score'] == 0
    api_status = 'danger' if api['vt_score'] > 50 or api['gsb_score'] > 50 else ('warn' if api_all_failed else 'ok')
    api_detail = (
        f'VirusTotal: {api["vt_score"]}  |  Google GSB: {api["gsb_score"]}  |  URLScan: {api["us_score"]}'
        if not api_all_failed
        else f'⚠️ All APIs unavailable — using heuristics + ML only  |  VT: {api["vt_msg"][:30]}'
    )
    pipeline_steps.append({
        'step': 'External API Checks',
        'detail': api_detail,
        'status': api_status
    })
    for m in (api['vt_msg'], api['gsb_msg'], api['us_msg']):
        if m and 'clean' not in m.lower() and 'unavailable' not in m.lower() and 'error' not in m.lower():
            signals.append(m)
    if api['vt_score'] > 0:
        signals.append(api['vt_msg'])
    if api['gsb_score'] > 0:
        signals.append(api['gsb_msg'])

    # Step 7: ML Model (Random Forest)
    ml_score, shap_explanations, model_used = run_ml_prediction(url)
    pipeline_steps.append({
        'step': 'AI/ML Model (Random Forest — 15 features)',
        'detail': f'{ml_score:.0f}% phishing probability' + (' [model unavailable]' if not model_used else ''),
        'status': 'danger' if ml_score >= 70 else 'warn' if ml_score >= 40 else 'ok'
    })
    if ml_score > 0:
        signals.append(f'RF model: {ml_score:.0f}% phishing probability')

    # Step 8: SHAP Explanation
    if shap_explanations:
        top_reason = shap_explanations[0]['feature'] if shap_explanations else 'N/A'
        pipeline_steps.append({
            'step': 'SHAP Explanation',
            'detail': f'Top reason: {top_reason} | {len(shap_explanations)} features explained',
            'status': 'ok'
        })
    else:
        pipeline_steps.append({
            'step': 'SHAP Explanation',
            'detail': 'SHAP not available (model not loaded)',
            'status': 'warn'
        })

    # Step 9: Dynamic Risk Scoring (DRS)
    final_score = combine_scores(
        api['vt_score'], api['gsb_score'], api['us_score'],
        ml_score, heuristic_score, domain_age_days, False
    )
    pipeline_steps.append({
        'step': 'Dynamic Risk Scoring (DRS)',
        'detail': f'Final Risk Score: {final_score} / 100  (Blacklist + Domain Age + APIs + ML)',
        'status': 'ok'
    })

    # Step 10: Final Classification
    verdict, verdict_icon, verdict_cls = final_classify(final_score)
    pipeline_steps.append({
        'step': 'Final Classification',
        'detail': f'{verdict_icon} {verdict}',
        'status': verdict_cls if verdict_cls != 'safe' else 'ok'
    })

    return {
        'url': url, 'domain': domain, 'score': final_score, 'verdict': verdict,
        'verdict_icon': verdict_icon, 'verdict_cls': verdict_cls,
        'blacklisted': False, 'expanded': was_expanded, 'expanded_url': expanded_url if was_expanded else url,
        'domain_age_days': domain_age_days, 'domain_age_display': age_display,
        'vt_score': api['vt_score'], 'gsb_score': api['gsb_score'], 'us_score': api['us_score'],
        'ml_score': ml_score, 'model_loaded': _model_loaded,
        'signals': list(set(s for s in signals if s)),
        'shap_explanations': shap_explanations,
        'pipeline_steps': pipeline_steps,
        'heuristic_score': heuristic_score,
    }

def analyze_phishing_url(url):
    signals = []
    vt_score,  vt_msg  = _virustotal_check(url)
    gsb_score, gsb_msg = _google_safe_browsing(url)
    us_score,  us_msg  = _urlscan_check(url)
    for m in (vt_msg, gsb_msg, us_msg):
        if m: signals.append(m)

    # [F1] ML model
    ml_score = 0.0
    if _model_loaded and clf and scaler:
        try:
            X_scaled = scaler.transform([extract_url_features(url)])
            ml_score = round(float(clf.predict_proba(X_scaled)[0][1]) * 100, 1)
            signals.append(f"RF model: {ml_score:.0f}% phishing probability")
        except Exception as e:
            signals.append(f"ML error: {e}")

    # [F3] SHAP
    shap_explanations = []
    if shap_explainer is not None:
        try:
            feats    = extract_url_features(url)
            X_scaled = scaler.transform([feats])
            sv       = shap_explainer.shap_values(X_scaled)
            vals     = sv[1][0] if isinstance(sv, list) else sv[0]
            top5 = sorted(zip(FEATURE_NAMES, vals, feats), key=lambda x: abs(x[1]), reverse=True)[:5]
            shap_explanations = [{'feature': f, 'impact': round(float(v), 4), 'value': round(float(r), 2)}
                                 for f, v, r in top5]
        except Exception as e:
            shap_explanations = [{'feature': f'SHAP error: {e}', 'impact': 0, 'value': 0}]

    # [F4] DRS
    final_score = drs_score(vt_score, gsb_score, us_score, ml_score)

    p = urlparse(url); domain = p.netloc.lower()
    heuristic_score = 0
    if re.match(r'^\d+\.\d+\.\d+\.\d+$', domain.split(':')[0]):
        signals.append('IP address as domain'); heuristic_score += 40
    if '@' in url:
        signals.append('@ symbol in URL'); heuristic_score += 30

    if p.scheme != 'https':
        signals.append('No HTTPS'); heuristic_score += 20
    kw_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url.lower())
    if kw_hits >= 3:  # Raised threshold: 2→3 to avoid false positives on normal URLs
        signals.append(str(kw_hits) + ' suspicious keywords found')
        heuristic_score += min(kw_hits * 8, 30)

    # If external APIs returned nothing (0), combine ML + heuristics as fallback
    if vt_score == 0 and gsb_score == 0:
        fallback = int(ml_score * 0.5 + heuristic_score * 0.5)
        final_score = min(100, max(final_score, fallback))

    verdict = ('Phishing likely' if final_score >= 70 else 'Suspicious' if final_score >= 40 else 'Safe')
    return {'url': url, 'score': final_score, 'vt_score': vt_score, 'gsb_score': gsb_score,
            'us_score': us_score, 'ml_score': ml_score, 'verdict': verdict,
            'signals': signals, 'shap_explanations': shap_explanations, 'model_loaded': _model_loaded}


# ═══ [F2] EXE ANALYSIS — Real VirusTotal ══════════════════════════════════

def analyze_exe_with_vt(sha256, filename):
    if not VT_API_KEY:
        return {'verdict': 'Unknown', 'confidence': 0.0, 'vt_engines': 0,
                'malicious': 0, 'total': 0, 'source': 'no_api_key'}
    try:
        h = {"x-apikey": VT_API_KEY}
        r = http_requests.get(f"https://www.virustotal.com/api/v3/files/{sha256}",
                              headers=h, timeout=15)
        if r.status_code == 404:
            return {'verdict': 'Unknown (not in VirusTotal)', 'confidence': 0.0,
                    'vt_engines': 0, 'malicious': 0, 'total': 0, 'source': 'vt_not_found'}
        if r.status_code == 200:
            stats      = r.json()["data"]["attributes"]["last_analysis_stats"]
            malicious  = stats.get("malicious", 0)
            suspicious = stats.get("suspicious", 0)
            total      = sum(stats.values())
            confidence = round((malicious + suspicious * 0.5) / total, 3) if total else 0.0
            verdict    = 'Malicious' if confidence > 0.3 else 'Suspicious' if confidence > 0.1 else 'Safe'
            return {'verdict': verdict, 'confidence': confidence, 'vt_engines': malicious,
                    'malicious': malicious, 'suspicious': suspicious, 'total': total, 'source': 'virustotal'}
        return {'verdict': 'Unknown', 'confidence': 0.0, 'vt_engines': 0,
                'malicious': 0, 'total': 0, 'source': f'vt_error_{r.status_code}'}
    except Exception as e:
        return {'verdict': 'Unknown', 'confidence': 0.0, 'vt_engines': 0,
                'malicious': 0, 'total': 0, 'source': f'exception: {e}'}


# ═══ [F7] SOCKET.IO ════════════════════════════════════════════════════════

@socketio.on('join')
def on_join(data):
    room = data.get('room', '')
    if room: join_room(room)

def push_alert(user_id, alert_type, message, extra=None):
    IST = timezone(timedelta(hours=5, minutes=30))
    payload = {'type': alert_type, 'message': message,
                'timestamp': datetime.now(IST).strftime('%I:%M %p IST')}
    if extra: payload.update(extra)
    try: socketio.emit('alert', payload, room=f"user_{user_id}")
    except: pass


# ═══ EMAIL / OTP HELPERS ══════════════════════════════════════════════════

def send_otp_sms(mobile, otp):
    try:
        from twilio.rest import Client
        Client(os.environ.get('TWILIO_ACCOUNT_SID'),
               os.environ.get('TWILIO_AUTH_TOKEN')).messages.create(
            body=f"Your TrustCheck OTP is: {otp}. Valid for 5 minutes.",
            from_=os.environ.get('TWILIO_FROM_NUMBER'), to=f"+91{mobile}")
        return True
    except Exception as e:
        print(f"[TWILIO] {e}\n[FALLBACK] OTP for {mobile}: {otp}")
        return False

def send_otp_email(email, otp, name=''):
    try:
        msg = Message(subject="Your TrustCheck Login OTP", recipients=[email])
        msg.html = f"""<div style="font-family:Segoe UI,sans-serif;max-width:480px;margin:auto;padding:30px;border-radius:12px;border:1px solid #e0e0e0;">
            <h2 style="color:#6a11cb;">&#x1F6E1; TrustCheck</h2><p>Hello <b>{name}</b>,</p>
            <p>Your OTP:</p><div style="font-size:2.5em;font-weight:bold;letter-spacing:10px;color:#2575fc;text-align:center;padding:20px;background:#f0f4ff;border-radius:8px;margin:20px 0;">{otp}</div>
            <p style="color:#888;font-size:0.9em;">Valid for <b>5 minutes</b>.</p></div>"""
        mail.send(msg)
        return True
    except Exception as e:
        print(f"[EMAIL] {e}\n[FALLBACK] OTP for {email}: {otp}")
        return False


# ═══ AUTH ROUTES ══════════════════════════════════════════════════════════

@app.route('/')
def index(): return redirect(url_for('dashboard_landing'))

@app.route('/home')
def dashboard_landing(): return render_template('dashboard_landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        aadhaar_number = re.sub(r'\s+', '', request.form.get('aadhaar_number', ''))
        if not re.fullmatch(r'\d{12}', aadhaar_number):
            flash("Aadhaar must be exactly 12 digits.", 'danger'); return redirect(url_for('register'))
        if User.query.filter_by(aadhaar_number=aadhaar_number).first():
            flash("Account already exists.", 'danger'); return redirect(url_for('register'))
        rec = AadhaarRecord.query.filter_by(aadhaar_number=aadhaar_number).first()
        if not rec:
            flash("Aadhaar not found in database.", 'danger'); return redirect(url_for('register'))
        user = User(full_name=rec.name, state=rec.state, gender=rec.gender,
                    aadhaar_number=aadhaar_number, mobile_number=rec.mobile)
        db.session.add(user); db.session.commit()
        otp = generate_otp()
        user.otp = otp
        user.otp_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=5)
        db.session.commit(); send_otp_sms(rec.mobile, otp)
        session['reg_aadhaar'] = aadhaar_number
        flash(f"OTP sent to ****{rec.mobile[-4:]}.", 'info')
        return redirect(url_for('verify_otp'))
    return render_template('register.html')

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    aadhaar = session.get('reg_aadhaar')
    if not aadhaar: flash("No registration in progress.", 'warning'); return redirect(url_for('register'))
    user = User.query.filter_by(aadhaar_number=aadhaar).first()
    if request.method == 'POST':
        entered = request.form.get('otp', '').strip()
        if not user or user.otp != entered or datetime.now(timezone.utc).replace(tzinfo=None) > user.otp_expires_at:
            flash("Invalid or expired OTP.", 'danger'); return redirect(url_for('verify_otp'))
        user.otp = user.otp_expires_at = None; db.session.commit()
        session['reg_set_creds'] = aadhaar; session.pop('reg_aadhaar', None)
        return redirect(url_for('register_confirm'))
    masked = f"****{user.mobile_number[-4:]}" if user else "your mobile"
    return render_template('verify_otp.html', masked_mobile=masked,
                           resend_url=url_for('resend_reg_otp'), form_action=url_for('verify_otp'))

@app.route('/resend-reg-otp', methods=['POST'])
def resend_reg_otp():
    aadhaar = session.get('reg_aadhaar')
    user = User.query.filter_by(aadhaar_number=aadhaar).first() if aadhaar else None
    if not user: flash("Session expired.", 'warning'); return redirect(url_for('register'))
    otp = generate_otp(); user.otp = otp
    user.otp_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=5)
    db.session.commit(); send_otp_sms(user.mobile_number, otp)
    flash("New OTP sent!", 'success'); return redirect(url_for('verify_otp'))

@app.route('/send-email-otp', methods=['POST'])
def send_email_otp():
    aadhaar = session.get('reg_set_creds')
    if not aadhaar: return jsonify({'success': False, 'error': 'Session expired.'}), 400
    email = request.form.get('email', '').strip().lower()
    if not re.fullmatch(r'[^@]+@[^@]+\.[^@]+', email):
        return jsonify({'success': False, 'error': 'Invalid email.'}), 400
    user = User.query.filter_by(aadhaar_number=aadhaar).first()
    if not user: return jsonify({'success': False, 'error': 'User not found.'}), 400
    if User.query.filter(User.email == email, User.id != user.id).first():
        return jsonify({'success': False, 'error': 'Email already registered.'}), 400
    otp = generate_otp()
    session['reg_email_otp']        = otp
    session['reg_email_otp_target'] = email
    session['reg_email_otp_expires'] = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    sent = send_otp_email(email, otp, name=user.full_name)
    masked = f"{email[:3]}***{email[email.index('@'):]}"
    return jsonify({'success': True, 'masked': masked, 'dev_fallback': not sent})

@app.route('/register-confirm')
def register_confirm():
    aadhaar = session.get('reg_set_creds')
    if not aadhaar: flash('Session expired.', 'warning'); return redirect(url_for('register'))
    user = User.query.filter_by(aadhaar_number=aadhaar).first()
    if not user: return redirect(url_for('register'))
    return render_template('register_confirm.html', user=user)

@app.route('/set-credentials', methods=['GET', 'POST'])
def set_credentials():
    aadhaar = session.get('reg_set_creds')
    if not aadhaar: flash("Session expired.", 'warning'); return redirect(url_for('register'))
    user = User.query.filter_by(aadhaar_number=aadhaar).first()
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')
        entered_otp = request.form.get('email_otp', '').strip()
        errors = []
        if not re.fullmatch(r'[A-Za-z0-9_]{4,30}', username): errors.append("Username: 4–30 alphanumeric.")
        if User.query.filter(User.username == username, User.id != user.id).first(): errors.append("Username taken.")
        if not re.fullmatch(r'[^@]+@[^@]+\.[^@]+', email): errors.append("Invalid email.")
        if User.query.filter(User.email == email, User.id != user.id).first(): errors.append("Email registered.")
        if len(password) < 8: errors.append("Password min 8 chars.")
        if password != confirm: errors.append("Passwords don't match.")
        sess_otp = session.get('reg_email_otp'); sess_target = session.get('reg_email_otp_target')
        sess_exp = session.get('reg_email_otp_expires')
        if not sess_otp or not sess_target or not sess_exp: errors.append("Verify email first.")
        elif sess_target != email: errors.append("Email mismatch with OTP.")
        elif datetime.now(timezone.utc) > datetime.fromisoformat(sess_exp): errors.append("OTP expired.")
        elif entered_otp != sess_otp: errors.append("Wrong OTP.")
        if errors:
            for e in errors: flash(e, 'danger')
            return redirect(url_for('set_credentials'))
        for k in ('reg_email_otp','reg_email_otp_target','reg_email_otp_expires'): session.pop(k, None)
        user.username = username; user.email = email
        user.password_hash = generate_password_hash(password); user.is_verified = True
        # [F6] Generate honeywords
        honeys = [''.join(random.choices(string.ascii_letters + string.digits, k=12)) for _ in range(5)]
        user.honeywords = json.dumps([generate_password_hash(h) for h in honeys])
        db.session.commit(); session.pop('reg_set_creds', None)
        flash("Registration complete!", 'success'); return redirect(url_for('login'))
    return render_template('set_credentials.html', full_name=user.full_name if user else '')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not re.fullmatch(r'[^@]+@[^@]+\.[^@]+', email):
            flash("Invalid email.", 'danger'); return redirect(url_for('login'))
        user = User.query.filter_by(email=email, is_verified=True).first()
        # [F6] Honeyword check
        if user and not check_password_hash(user.password_hash, password) and user.honeywords:
            for hw_hash in json.loads(user.honeywords):
                if check_password_hash(hw_hash, password):
                    db.session.add(HoneywordAttempt(user_id=user.id, ip_address=request.remote_addr))
                    db.session.commit()
                    push_alert(user.id, 'honeyword',
                               f'CREDENTIAL THEFT DETECTED from IP {request.remote_addr}. Change your password!',
                               {'ip': request.remote_addr})
                    print(f"[HONEYWORD] Theft detected for {email} from {request.remote_addr}")
                    break
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", 'danger'); return redirect(url_for('login'))
        otp = generate_otp(); user.otp = otp
        user.otp_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=5)
        db.session.commit()
        sent = send_otp_email(email, otp, name=user.full_name)
        masked = f"{email[:3]}***{email[email.index('@'):]}"
        flash(f"OTP sent to {masked}", 'success' if sent else 'warning')
        session['login_aadhaar'] = user.aadhaar_number
        return redirect(url_for('login_verify_otp'))
    return render_template('login.html')

@app.route('/login-verify-otp', methods=['GET', 'POST'])
def login_verify_otp():
    aadhaar = session.get('login_aadhaar')
    if not aadhaar: flash("No login in progress.", 'warning'); return redirect(url_for('login'))
    user = User.query.filter_by(aadhaar_number=aadhaar).first()
    if request.method == 'POST':
        entered = request.form.get('otp', '').strip()
        if not user or user.otp != entered or datetime.now(timezone.utc).replace(tzinfo=None) > user.otp_expires_at:
            flash("Invalid or expired OTP.", 'danger'); return redirect(url_for('login_verify_otp'))
        user.otp = user.otp_expires_at = None; db.session.commit()
        session.update({'user_id': user.id, 'username': user.username,
                        'full_name': user.full_name, 'email': user.email})
        session.pop('login_aadhaar', None)
        log_session_event(user.id, 'login')  # [F5]
        anomaly = detect_behavioral_anomaly(user.id)  # [F5]
        if anomaly['anomaly']:
            push_alert(user.id, 'anomaly', f'Unusual login: {anomaly["reason"]}',
                       {'score': anomaly['score']})
        flash(f"Welcome back, {user.full_name}!", 'success')
        return redirect(url_for('dashboard'))
    masked = f"{user.email[:3]}***{user.email[user.email.index('@'):]}" if user and user.email else "your email"
    return render_template('login_otp.html', masked_email=masked,
                           resend_url=url_for('resend_login_otp'), form_action=url_for('login_verify_otp'))

@app.route('/resend-login-otp', methods=['POST'])
def resend_login_otp():
    aadhaar = session.get('login_aadhaar')
    user = User.query.filter_by(aadhaar_number=aadhaar).first() if aadhaar else None
    if not user: flash("Session expired.", 'warning'); return redirect(url_for('login'))
    otp = generate_otp(); user.otp = otp
    user.otp_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=5)
    db.session.commit(); send_otp_email(user.email, otp, name=user.full_name)
    flash("New OTP sent!", 'success')
    masked = f"{user.email[:3]}***{user.email[user.email.index('@'):]}"
    return render_template('login_otp.html', masked_email=masked,
                           resend_url=url_for('resend_login_otp'), form_action=url_for('login_verify_otp'))

@app.route('/logout')
def logout():
    session.clear(); flash("Logged out.", 'info'); return redirect(url_for('dashboard_landing'))


# ═══ DASHBOARD ═════════════════════════════════════════════════════════════

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: flash("Please log in.", 'warning'); return redirect(url_for('login'))
    uid = session['user_id']
    urls_scanned   = PhishingHistory.query.filter_by(user_id=uid).count()
    phishing_found = PhishingHistory.query.filter(PhishingHistory.user_id==uid,
                                                   PhishingHistory.status.ilike('%phish%')).count()
    safe_count     = PhishingHistory.query.filter(PhishingHistory.user_id==uid,
                                                   PhishingHistory.status.ilike('%safe%')).count()
    recent_phishing = PhishingHistory.query.filter_by(user_id=uid)\
        .order_by(PhishingHistory.created_at.desc()).limit(10).all()
    for r in recent_phishing: r.url_display = decrypt_text(r.url)
    procs = []
    for p in psutil.process_iter(['pid','name','cpu_percent','memory_percent']):
        try: procs.append({'pid':p.info['pid'],'name':p.info['name'],'cpu':p.info['cpu_percent'],'memory':p.info['memory_percent']})
        except psutil.NoSuchProcess: pass
    top_procs = sorted(procs, key=lambda x: x['cpu'], reverse=True)[:8]
    caller_records = CallerHistory.query.filter_by(user_id=uid).order_by(CallerHistory.checked_at.desc()).all()
    IST = timezone(timedelta(hours=5, minutes=30))
    history = [{'number':r.number,'name':r.name,'verified':r.verified,'gender':r.gender,
                'checked_at':r.checked_at.replace(tzinfo=timezone.utc).astimezone(IST).strftime('%d %b %Y, %I:%M %p IST')} for r in caller_records]
    last_caller = None
    if caller_records:
        r = caller_records[0]
        last_caller = {'number':r.number,'received_at':r.checked_at.replace(tzinfo=timezone.utc).astimezone(IST).strftime('%d %b %Y, %I:%M %p IST')}
    return render_template('dashboard.html',
        urls_scanned=urls_scanned, phishing_found=phishing_found, safe_count=safe_count,
        recent_phishing=recent_phishing, total_procs=len(procs), top_procs=top_procs,
        history=history, last_caller=last_caller,
        callers_verified=CallerHistory.query.filter_by(user_id=uid,verified=True).count(),
        honeyword_alerts=HoneywordAttempt.query.filter_by(user_id=uid).count(),
        model_loaded=_model_loaded)

@app.route('/phishing-checker')
def phishing_checker():
    if 'user_id' not in session: flash("Please log in.", 'warning'); return redirect(url_for('login'))
    return render_template('phishing_checker.html', model_loaded=_model_loaded)

@app.route('/phishing-check', methods=['POST'])
def phishing_check():
    url    = request.json.get('url', '')
    use_pipeline = request.json.get('pipeline', True)
    result = run_pipeline(url) if use_pipeline else analyze_phishing_url(url)
    if 'user_id' in session:
        db.session.add(PhishingHistory(
            user_id=session['user_id'], url=encrypt_text(url),  # [F8]
            status=result['verdict'], confidence=result['score'],
            ml_score=result['ml_score'], drs_score_val=result['score'],
            shap_top5=json.dumps(result['shap_explanations'])))  # [F3]
        db.session.commit()
        if result['score'] >= 60:  # [F7]
            push_alert(session['user_id'], 'phishing',
                       f"Phishing detected: {url[:60]}... (Risk: {result['score']}%)",
                       {'url': url, 'score': result['score'], 'verdict': result['verdict']})
        log_session_event(session['user_id'], 'phishing_scan')  # [F5]
    return jsonify(result)

@app.route('/phishing-history')
def phishing_history():
    if 'user_id' not in session: flash("Please log in.", 'warning'); return redirect(url_for('login'))
    records = PhishingHistory.query.filter_by(user_id=session['user_id'])\
                .order_by(PhishingHistory.created_at.desc()).all()
    data = [{'url': decrypt_text(r.url), 'status': r.status, 'confidence': r.confidence,
              'ml_score': r.ml_score, 'created_at': r.created_at.strftime('%Y-%m-%d %H:%M:%S'),
              'shap_top5': json.loads(r.shap_top5) if r.shap_top5 else []} for r in records]
    return render_template('phishing_history.html', history=data)


# ═══ EXE ANALYSIS — [F2] ══════════════════════════════════════════════════

@app.route('/analyze-exe-page')
def analyze_exe_page(): return render_template('analyze_exe.html')

@app.route('/analyze-exe', methods=['POST'])
def analyze_exe():
    f = request.files.get('file')
    if not f or not f.filename.lower().endswith('.exe'):
        return jsonify({'status': 'error', 'message': 'Only .exe files allowed'})
    content = f.read(); sha256 = hashlib.sha256(content).hexdigest()
    vt = analyze_exe_with_vt(sha256, f.filename)
    if 'user_id' in session:
        db.session.add(ExeHistory(user_id=session['user_id'], exe_name=f.filename,
            hash=sha256, verdict=vt['verdict'],
            confidence=f"{vt['confidence']*100:.1f}%", vt_engines=vt['vt_engines']))
        db.session.commit()
        if vt['verdict'] in ('Malicious', 'Suspicious'):
            push_alert(session['user_id'], 'malware',
                       f"{vt['verdict']} file: {f.filename} ({vt['vt_engines']} VT engines)",
                       {'filename': f.filename, 'hash': sha256})
        log_session_event(session['user_id'], 'exe_upload')
    return jsonify({'status':'success','verdict':vt['verdict'],'confidence':vt['confidence'],
                    'hash':sha256,'vt_engines':vt['vt_engines'],'vt_total':vt.get('total',0),
                    'source':vt['source'],'malicious':vt.get('malicious',0),'suspicious':vt.get('suspicious',0)})

@app.route('/exe-history')
def exe_history():
    if 'user_id' not in session: flash("Please log in.", 'warning'); return redirect(url_for('login'))
    records = ExeHistory.query.filter_by(user_id=session['user_id'])\
                .order_by(ExeHistory.created_at.desc()).all()
    data = [{'exe_name':r.exe_name,'hash':r.hash,'verdict':r.verdict,'confidence':r.confidence,
              'vt_engines':r.vt_engines,'created_at':r.created_at.strftime('%Y-%m-%d %H:%M:%S')} for r in records]
    return render_template('exe_history.html', history=data)


# ═══ PROCESS MONITOR ══════════════════════════════════════════════════════

@app.route('/apps-monitor')
def apps_monitor():
    procs = []
    for p in psutil.process_iter(['pid','name','cpu_percent','memory_percent']):
        try: procs.append({'pid':p.info['pid'],'name':p.info['name'],'cpu':p.info['cpu_percent'],'memory':p.info['memory_percent']})
        except psutil.NoSuchProcess: pass
    if 'user_id' in session:
        for p in procs[:50]:
            db.session.add(ProcessHistory(user_id=session['user_id'],pid=p['pid'],name=p['name'],cpu=p['cpu'],memory=p['memory']))
        db.session.commit()
    return render_template('apps_monitor.html', processes=procs)

@app.route('/get-processes')
def get_processes():
    procs = []
    for p in psutil.process_iter(['pid','name','cpu_percent','memory_percent']):
        try: procs.append({'pid':p.info['pid'],'name':p.info['name'],'cpu':p.info['cpu_percent'],'memory':p.info['memory_percent']})
        except psutil.NoSuchProcess: pass
    return jsonify(procs)

@app.route('/kill-process/<int:pid>', methods=['POST'])
def kill_process(pid):
    try: os.kill(pid, signal.SIGTERM); return jsonify({'status':'success','message':f'Process {pid} terminated'})
    except Exception as e: return jsonify({'status':'error','message':str(e)})

@app.route('/process-history')
def process_history():
    if 'user_id' not in session: flash("Please log in.", 'warning'); return redirect(url_for('login'))
    records = ProcessHistory.query.filter_by(user_id=session['user_id'])\
                .order_by(ProcessHistory.created_at.desc()).limit(50).all()
    data = [{'pid':r.pid,'name':r.name,'cpu':r.cpu,'memory':r.memory,
              'created_at':r.created_at.strftime('%Y-%m-%d %H:%M:%S')} for r in records]
    return render_template('process_history.html', history=data)


# ═══ CALLER VERIFY ════════════════════════════════════════════════════════

@app.route('/api/verify-caller', methods=['POST'])
def api_verify_caller():
    if 'user_id' not in session: return jsonify({'verified':False,'message':'Not logged in'}), 401
    number = normalize_mobile(request.json.get('number','').strip())
    rec = AadhaarRecord.query.filter_by(mobile=number).first()
    verified = rec is not None
    db.session.add(CallerHistory(user_id=session['user_id'],number=number,
        name=rec.name if rec else None,verified=verified,gender=rec.gender if rec else None))
    db.session.commit()
    if verified: return jsonify({'verified':True,'name':rec.name,'gender':rec.gender,'state':rec.state})
    return jsonify({'verified':False,'message':'No Aadhaar record found.'})

@app.route('/caller-history')
def caller_history():
    if 'user_id' not in session: flash("Please log in.", 'warning'); return redirect(url_for('login'))
    records = CallerHistory.query.filter_by(user_id=session['user_id'])\
                .order_by(CallerHistory.checked_at.desc()).all()
    IST = timezone(timedelta(hours=5, minutes=30))
    return jsonify([{'number':r.number,'name':r.name or '—','verified':r.verified,
                     'gender':r.gender or '—','checked_at':r.checked_at.replace(tzinfo=timezone.utc).astimezone(IST).strftime('%d %b %Y, %I:%M %p IST')} for r in records])

@app.route('/caller-verify')
def caller_verify():
    if 'user_id' not in session: flash("Please log in.", 'warning'); return redirect(url_for('login'))
    uid = session['user_id']
    records = CallerHistory.query.filter_by(user_id=uid).order_by(CallerHistory.checked_at.desc()).all()
    IST = timezone(timedelta(hours=5, minutes=30))
    history = [{'number':r.number,'name':r.name,'verified':r.verified,'gender':r.gender,
                'checked_at':r.checked_at.replace(tzinfo=timezone.utc).astimezone(IST).strftime('%d %b %Y, %I:%M %p IST')} for r in records]
    last_caller = None
    if records:
        r = records[0]; last_caller = {'number':r.number,'received_at':r.checked_at.replace(tzinfo=timezone.utc).astimezone(IST).strftime('%d %b %Y, %I:%M %p IST')}
    device_online = False
    try:
        dev = DeviceStatus.query.filter_by(user_id=uid).order_by(DeviceStatus.updated_at.desc()).first()
        if dev:
            age = (datetime.now(timezone.utc).replace(tzinfo=None) - dev.updated_at).total_seconds()
            device_online = (dev.status == 'online' and age < 120)
    except: pass
    return render_template('caller_verify.html', history=history,
                           last_caller=last_caller, device_online=device_online)

@app.route('/api/last-caller')
def api_last_caller():
    if 'user_id' not in session: return jsonify({'error':'not logged in'}), 401
    r = CallerHistory.query.filter_by(user_id=session['user_id']).order_by(CallerHistory.checked_at.desc()).first()
    if not r: return jsonify({'number':None})
    IST = timezone(timedelta(hours=5, minutes=30))
    return jsonify({'number':r.number,'name':r.name,'verified':r.verified,'gender':r.gender,
                    'received_at':r.checked_at.replace(tzinfo=timezone.utc).astimezone(IST).strftime('%d %b %Y, %I:%M %p IST')})

@app.route('/api/device-status-check')
def api_device_status_check():
    if 'user_id' not in session: return jsonify({'error':'not logged in'}), 401
    try:
        r = DeviceStatus.query.filter_by(user_id=session['user_id']).order_by(DeviceStatus.updated_at.desc()).first()
        if not r: return jsonify({'status':'offline','device_id':None,'updated_at':None})
        age = (datetime.now(timezone.utc).replace(tzinfo=None) - r.updated_at).total_seconds()
        IST = timezone(timedelta(hours=5, minutes=30))
        return jsonify({'status': r.status if age < 120 else 'offline',
                        'device_id':r.device_id,'updated_at':r.updated_at.replace(tzinfo=timezone.utc).astimezone(IST).strftime('%d %b %Y, %I:%M %p IST')})
    except: return jsonify({'status':'offline','device_id':None,'updated_at':None})


# ═══ SECURITY STATUS ══════════════════════════════════════════════════════

@app.route('/api/security-status')
def api_security_status():
    if 'user_id' not in session: return jsonify({'error':'not logged in'}), 401
    uid = session['user_id']
    honeyword_hits = HoneywordAttempt.query.filter_by(user_id=uid)\
                        .order_by(HoneywordAttempt.attempted_at.desc()).limit(10).all()
    return jsonify({
        'honeyword_attempts': [{'ip':h.ip_address,'at':h.attempted_at.strftime('%Y-%m-%d %H:%M:%S')} for h in honeyword_hits],
        'model_loaded': _model_loaded, 'encryption': 'AES-256 Fernet',
        'shap_active': shap_explainer is not None,
    })

@app.route('/dashboard-stats')
def dashboard_stats():
    if 'user_id' not in session: return jsonify({'error':'not logged in'}), 401
    uid = session['user_id']
    procs = []
    for p in psutil.process_iter(['pid','name','cpu_percent','memory_percent']):
        try: procs.append(p.info)
        except psutil.NoSuchProcess: pass
    return jsonify({'urls_scanned': PhishingHistory.query.filter_by(user_id=uid).count(),
                    'phishing_found': PhishingHistory.query.filter(PhishingHistory.user_id==uid,
                        PhishingHistory.status.ilike('%phish%')).count(),
                    'total_procs': len(procs),
                    'honeyword_alerts': HoneywordAttempt.query.filter_by(user_id=uid).count(),
                    'model_loaded': _model_loaded})


# ═══ LOG ANALYSIS ═════════════════════════════════════════════════════════

def analyze_log_content(content):
    failed_ips, failed_users, timeline = {}, {}, []
    for line in content.splitlines():
        if "Failed password" in line:
            ip = re.search(r'from (\d+\.\d+\.\d+\.\d+)', line)
            user = re.search(r'Failed password for (\w+)', line)
            ts = re.search(r'^(\w+\s+\d+\s+\d+:\d+:\d+)', line)
            if ip:
                ip = ip.group(1); failed_ips[ip] = failed_ips.get(ip,0)+1
                if user and ts:
                    u = user.group(1); failed_users[u] = failed_users.get(u,0)+1
                    timeline.append({'timestamp':ts.group(1),'ip':ip,'user':u})
    return {'ip_analysis':failed_ips,'user_analysis':failed_users,'timeline':timeline}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({'error':'No file'}), 400
    f = request.files['file']
    if f.filename == '' or not allowed_file(f.filename): return jsonify({'error':'Invalid file'}), 400
    return jsonify(analyze_log_content(f.read().decode('utf-8', errors='ignore')))


# ═══ HISTORY JSON APIs ════════════════════════════════════════════════════

@app.route('/api/history/phishing')
def api_phishing_history():
    if 'user_id' not in session: return jsonify([])
    records = PhishingHistory.query.filter_by(user_id=session['user_id'])\
                .order_by(PhishingHistory.created_at.desc()).all()
    return jsonify([{'url':decrypt_text(r.url),'status':r.status,'confidence':r.confidence,
                     'ml_score':r.ml_score,'created_at':r.created_at.strftime('%Y-%m-%d %H:%M')} for r in records])

@app.route('/api/history/exe')
def api_exe_history():
    if 'user_id' not in session: return jsonify([])
    records = ExeHistory.query.filter_by(user_id=session['user_id'])\
                .order_by(ExeHistory.created_at.desc()).all()
    return jsonify([{'exe_name':r.exe_name,'hash':r.hash,'verdict':r.verdict,'confidence':r.confidence,
                     'vt_engines':r.vt_engines,'created_at':r.created_at.strftime('%Y-%m-%d %H:%M')} for r in records])

@app.route('/api/history/processes')
def api_process_history():
    if 'user_id' not in session: return jsonify([])
    records = ProcessHistory.query.filter_by(user_id=session['user_id'])\
                .order_by(ProcessHistory.created_at.desc()).limit(50).all()
    return jsonify([{'pid':r.pid,'name':r.name,'cpu':r.cpu,'memory':r.memory,
                     'created_at':r.created_at.strftime('%Y-%m-%d %H:%M')} for r in records])


# ═══ PDF EXPORTS ══════════════════════════════════════════════════════════

def _make_table_style():
    return TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1a1f3a')),
        ('TEXTCOLOR', (0,0),(-1,0),colors.white),
        ('FONTSIZE',  (0,0),(-1,0),9),('FONTSIZE',(0,1),(-1,-1),8),
        ('GRID',      (0,0),(-1,-1),0.4,colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7ff')]),
        ('VALIGN',    (0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
    ])

@app.route('/api/history/phishing/pdf')
def api_phishing_history_pdf():
    if 'user_id' not in session: return redirect(url_for('login'))
    records = PhishingHistory.query.filter_by(user_id=session['user_id'])\
                .order_by(PhishingHistory.created_at.desc()).all()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf,pagesize=A4,leftMargin=1.5*cm,rightMargin=1.5*cm,topMargin=2*cm,bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    elems = [Paragraph("TrustCheck v2 — Phishing History (AES-256 Encrypted)", styles['Title']), Spacer(1,0.4*cm)]
    data = [['#','URL','Verdict','DRS Score','ML Score','Timestamp']]
    for i,r in enumerate(records,1):
        u = decrypt_text(r.url)
        data.append([str(i),u[:45]+'…' if len(u)>45 else u,r.status,str(r.confidence or '—'),
                     f"{r.ml_score:.0f}%" if r.ml_score is not None else '—',r.created_at.strftime('%Y-%m-%d %H:%M')])
    t = Table(data,colWidths=[0.8*cm,5.5*cm,3*cm,2*cm,2*cm,3.5*cm])
    t.setStyle(_make_table_style()); elems.append(t); doc.build(elems); buf.seek(0)
    return send_file(buf,mimetype='application/pdf',download_name='phishing_history_v2.pdf',as_attachment=True)

@app.route('/api/history/exe/pdf')
def api_exe_history_pdf():
    if 'user_id' not in session: return redirect(url_for('login'))
    records = ExeHistory.query.filter_by(user_id=session['user_id'])\
                .order_by(ExeHistory.created_at.desc()).all()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf,pagesize=A4,leftMargin=1.5*cm,rightMargin=1.5*cm,topMargin=2*cm,bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    elems = [Paragraph("TrustCheck v2 — EXE History (Real VT Detection)", styles['Title']), Spacer(1,0.4*cm)]
    data = [['#','File Name','Verdict','Confidence','VT Engines','SHA256','Timestamp']]
    for i,r in enumerate(records,1):
        data.append([str(i),r.exe_name[:25]+'…' if len(r.exe_name)>25 else r.exe_name,
                     r.verdict,r.confidence or '—',str(r.vt_engines) if r.vt_engines is not None else '—',
                     r.hash[:14]+'…' if r.hash else '—',r.created_at.strftime('%Y-%m-%d %H:%M')])
    t = Table(data,colWidths=[0.8*cm,3.5*cm,2.2*cm,2*cm,2*cm,2.5*cm,3*cm])
    t.setStyle(_make_table_style()); elems.append(t); doc.build(elems); buf.seek(0)
    return send_file(buf,mimetype='application/pdf',download_name='exe_history_v2.pdf',as_attachment=True)

@app.route('/api/history/processes/pdf')
def api_process_history_pdf():
    if 'user_id' not in session: return redirect(url_for('login'))
    records = ProcessHistory.query.filter_by(user_id=session['user_id'])\
                .order_by(ProcessHistory.created_at.desc()).limit(50).all()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf,pagesize=A4,leftMargin=1.5*cm,rightMargin=1.5*cm,topMargin=2*cm,bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    elems = [Paragraph("TrustCheck v2 — Process History", styles['Title']), Spacer(1,0.4*cm)]
    data = [['#','PID','Process Name','CPU %','Memory %','Timestamp']]
    for i,r in enumerate(records,1):
        data.append([str(i),str(r.pid),r.name[:35]+'…' if len(r.name)>35 else r.name,
                     f'{r.cpu}%',f'{r.memory}%',r.created_at.strftime('%Y-%m-%d %H:%M')])
    t = Table(data,colWidths=[0.8*cm,1.8*cm,5*cm,2*cm,2.2*cm,4*cm])
    t.setStyle(_make_table_style()); elems.append(t); doc.build(elems); buf.seek(0)
    return send_file(buf,mimetype='application/pdf',download_name='process_history_v2.pdf',as_attachment=True)

@app.route('/download-phishing-pdf')
def download_phishing_pdf(): return redirect(url_for('api_phishing_history_pdf'))
@app.route('/download-process-history-pdf')
def download_process_history_pdf(): return redirect(url_for('api_process_history_pdf'))
@app.route('/download-exe-history-pdf')
def download_exe_history_pdf(): return redirect(url_for('api_exe_history_pdf'))


# ═══ ADMIN PANEL ══════════════════════════════════════════════════════════

ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'Mohammed')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'Mohammed@123')

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash("Admin login required.", 'warning'); return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if (request.form.get('username','').strip() == ADMIN_USERNAME and
                request.form.get('password','').strip() == ADMIN_PASSWORD):
            session['admin_logged_in'] = True; flash("Welcome, Mohammed!", 'success')
            return redirect(url_for('admin_panel'))
        flash("Invalid admin credentials.", 'danger')
    return render_template('admin_login.html')

@app.route('/admin/panel')
@admin_required
def admin_panel():
    users = User.query.order_by(User.created_at.desc()).all()
    records = AadhaarRecord.query.order_by(AadhaarRecord.id).all()
    return render_template('admin_panel.html', users=users, aadhaar_records=records)

@app.route('/admin/aadhaar')
def admin_aadhaar():
    if 'user_id' not in session: flash("Please log in.", 'warning'); return redirect(url_for('login'))
    records = AadhaarRecord.query.order_by(AadhaarRecord.id).all()
    return render_template('admin_aadhaar.html', records=records)

@app.route('/admin/aadhaar/add', methods=['POST'])
def admin_aadhaar_add():
    if not session.get('admin_logged_in'): return redirect(url_for('admin_login'))
    aadhaar = re.sub(r'\s+','',request.form.get('aadhaar_number',''))
    name   = request.form.get('name','').strip()
    mobile = (request.form.get('mobile') or request.form.get('mobile_number') or '').strip()
    state  = request.form.get('state','').strip()
    gender = request.form.get('gender','').strip()
    if not all([aadhaar,name,mobile,state,gender]):
        flash("All fields required.", 'danger'); return redirect(url_for('admin_panel'))
    if not re.fullmatch(r'\d{12}',aadhaar):
        flash("Aadhaar must be 12 digits.", 'danger'); return redirect(url_for('admin_panel'))
    if not re.fullmatch(r'[6-9]\d{9}',mobile):
        flash("Invalid mobile number.", 'danger'); return redirect(url_for('admin_panel'))
    if AadhaarRecord.query.filter_by(aadhaar_number=aadhaar).first():
        flash("Aadhaar already exists.", 'danger'); return redirect(url_for('admin_panel'))
    try:
        db.session.add(AadhaarRecord(aadhaar_number=aadhaar,name=name,mobile=mobile,state=state,gender=gender))
        db.session.commit(); flash(f"Record for {name} added.", 'success')
    except Exception as e:
        db.session.rollback(); flash(f"DB error: {e}", 'danger')
    return redirect(url_for('admin_panel'))

@app.route('/admin/aadhaar/delete/<int:record_id>', methods=['POST'])
def admin_aadhaar_delete(record_id):
    if not session.get('admin_logged_in'): return redirect(url_for('admin_login'))
    rec = db.session.get(AadhaarRecord, record_id)
    if not rec: flash("Record not found.", 'danger'); return redirect(url_for('admin_panel'))
    name = rec.name; db.session.delete(rec); db.session.commit()
    flash(f"Record for {name} deleted.", 'info')
    return redirect(url_for('admin_panel'))

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in',None); flash("Admin logged out.", 'info')
    return redirect(url_for('admin_login'))


# ═══ ANDROID APP ENDPOINTS ════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({'status':'ok','message':'TrustCheck v2 running',
                    'model_loaded':_model_loaded,'encryption':'AES-256 Fernet',
                    'shap_active':shap_explainer is not None}), 200

@app.route('/api/login', methods=['POST'])
def api_mobile_login():
    data = request.get_json(silent=True)
    if not data: return jsonify({'error':'Invalid JSON'}), 400
    email = data.get('email','').strip().lower(); password = data.get('password','')
    if not email or not password: return jsonify({'error':'Email and password required'}), 400
    user = User.query.filter_by(email=email, is_verified=True).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'error':'Invalid credentials'}), 401
    import secrets; token = secrets.token_hex(32)
    user.otp = token
    user.otp_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=30)
    db.session.commit()
    return jsonify({'token':token,'user_id':str(user.id),'email':user.email,'name':user.full_name}), 200

def _get_user_from_token():
    auth = request.headers.get('Authorization',''); token = auth.replace('Bearer ','').strip()
    if not token: return None
    return User.query.filter(User.otp==token,
        User.otp_expires_at>=datetime.now(timezone.utc).replace(tzinfo=None)).first()

@app.route('/api/receive-number', methods=['POST'])
def api_receive_number():
    user = _get_user_from_token()
    if not user: return jsonify({'error':'Unauthorized'}), 401
    data = request.get_json(silent=True)
    if not data: return jsonify({'error':'Invalid JSON'}), 400
    phone = normalize_mobile(data.get('phone_number','').strip())
    if not phone: return jsonify({'error':'phone_number required'}), 400
    rec = AadhaarRecord.query.filter_by(mobile=phone).first(); verified = rec is not None
    db.session.add(CallerHistory(user_id=user.id,number=phone,
        name=rec.name if rec else None,verified=verified,gender=rec.gender if rec else None))
    db.session.commit()
    return jsonify({'status':'received','phone_number':phone,'call_type':data.get('call_type','UNKNOWN'),
                    'duration_sec':data.get('duration_sec',0),'verified':verified,
                    'name':rec.name if rec else None}), 200

@app.route('/api/device-status', methods=['POST'])
def api_device_status():
    user = _get_user_from_token()
    if not user: return jsonify({'error':'Unauthorized'}), 401
    data = request.get_json(silent=True) or {}
    device_id = data.get('device_id','unknown'); status = data.get('status','unknown')
    try:
        r = DeviceStatus.query.filter_by(user_id=user.id,device_id=device_id).first()
        if r: r.status = status; r.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        else:
            r = DeviceStatus(user_id=user.id,device_id=device_id,status=status,
                             updated_at=datetime.now(timezone.utc).replace(tzinfo=None))
            db.session.add(r)
        db.session.commit()
    except: db.session.rollback()
    return jsonify({'status':'ok','received':status}), 200

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'user_id' not in session: flash("Login first.", 'warning'); return redirect(url_for('login'))
    if request.method == 'POST' and clf and lookup_df is not None:
        number = request.form['number'].strip()
        row = lookup_df[lookup_df['number'].astype(str)==str(number)]
        X = scaler.transform([featurize_number(number)])
        pred = int(clf.predict(X)[0])
        return render_template('result.html', number=number, lookup=None if row.empty else row.iloc[0].to_dict(), pred=pred)
    return render_template('predict.html')


# ═══ ENTRY POINT ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  TrustCheck v2 — Security Platform")
    print("="*60)
    print(f"  [F1] ML Model:         {'LOADED' if _model_loaded else 'MISSING — run train_model.py'}")
    print(f"  [F2] EXE VT Analysis:  REAL VirusTotal hash lookup")
    print(f"  [F3] SHAP:             {'ACTIVE' if shap_explainer else 'needs model'}")
    print(f"  [F4] DRS Engine:       w1*Pa + w2*Ts + w3*Ac + w4*Ca")
    print(f"  [F5] Isolation Forest: Behavioral anomaly detection")
    print(f"  [F6] Honeywords:       Credential theft detection")
    print(f"  [F7] SocketIO:         Real-time WebSocket alerts")
    print(f"  [F8] Fernet Encrypt:   AES-256 scan result encryption")
    print("="*60 + "\n")
    port = int(os.environ.get('PORT', 5000))
    import socket
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = 'YOUR_PC_IP'
    print("")
    print("  >>> Web browser:  http://127.0.0.1:" + str(port))
    print("  >>> Android APK:  http://" + local_ip + ":" + str(port))
    print("  (phone and PC must be on same WiFi)")
    print("")
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=True)