"""Dummy payment gateway for demo subscriptions (not real payments)."""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime


PLAN_PRICES = {
    'pro': 9.99,
    'team': 29.99,
}

# Test card numbers — Stripe-style demo behavior
TEST_CARDS = {
    '4242424242424242': 'success',
    '4000000000000002': 'declined',
    '4000000000009995': 'expired',
}


@dataclass
class PaymentResult:
    success: bool
    transaction_id: str
    message: str
    status: str
    card_last4: str = ''


def _digits_only(value: str) -> str:
    return re.sub(r'\D', '', value or '')


def _luhn_check(card_number: str) -> bool:
    digits = [int(d) for d in card_number]
    checksum = 0
    reverse = digits[::-1]
    for i, digit in enumerate(reverse):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def validate_card(name: str, card_number: str, expiry: str, cvv: str) -> tuple[bool, str]:
    name = (name or '').strip()
    if len(name) < 2:
        return False, 'Please enter the name on card.'

    card_digits = _digits_only(card_number)
    if len(card_digits) != 16:
        return False, 'Card number must be 16 digits.'
    if not _luhn_check(card_digits):
        return False, 'Invalid card number.'

    expiry = (expiry or '').strip()
    if not re.match(r'^\d{2}/\d{2}$', expiry):
        return False, 'Expiry must be in MM/YY format.'

    month, year = expiry.split('/')
    month_int = int(month)
    year_int = 2000 + int(year)
    if month_int < 1 or month_int > 12:
        return False, 'Invalid expiry month.'

    now = datetime.now()
    if year_int < now.year or (year_int == now.year and month_int < now.month):
        return False, 'Card has expired.'

    cvv_digits = _digits_only(cvv)
    if len(cvv_digits) not in (3, 4):
        return False, 'CVV must be 3 or 4 digits.'

    return True, ''


def process_payment(plan: str, name: str, card_number: str, expiry: str, cvv: str) -> PaymentResult:
    if plan not in PLAN_PRICES:
        return PaymentResult(
            success=False,
            transaction_id='',
            message='Invalid subscription plan.',
            status='failed',
        )

    valid, error = validate_card(name, card_number, expiry, cvv)
    if not valid:
        return PaymentResult(
            success=False,
            transaction_id=f'TXN-{uuid.uuid4().hex[:12].upper()}',
            message=error,
            status='failed',
        )

    card_digits = _digits_only(card_number)
    card_last4 = card_digits[-4:]
    transaction_id = f'TXN-{uuid.uuid4().hex[:12].upper()}'

    outcome = TEST_CARDS.get(card_digits)
    if outcome is None and card_digits.startswith('4242'):
        outcome = 'success'

    if outcome == 'success':
        return PaymentResult(
            success=True,
            transaction_id=transaction_id,
            message='Payment approved.',
            status='completed',
            card_last4=card_last4,
        )

    if outcome == 'declined':
        return PaymentResult(
            success=False,
            transaction_id=transaction_id,
            message='Your card was declined. Please try another card.',
            status='declined',
            card_last4=card_last4,
        )

    if outcome == 'expired':
        return PaymentResult(
            success=False,
            transaction_id=transaction_id,
            message='This card has expired.',
            status='expired',
            card_last4=card_last4,
        )

    return PaymentResult(
        success=False,
        transaction_id=transaction_id,
        message='Unsupported test card. Use 4242 4242 4242 4242 for success.',
        status='declined',
        card_last4=card_last4,
    )
